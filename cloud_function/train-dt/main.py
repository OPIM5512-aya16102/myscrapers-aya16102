import os, io, json, logging, traceback
import numpy as np
import pandas as pd
from google.cloud import storage

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

import joblib
from scipy.stats import randint

# ---------------- ENV ----------------
PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
DATA_KEY = os.getenv("DATA_KEY", "structured_v2/datasets/listings_master_llm.csv")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "preds")

logging.basicConfig(level="INFO")

# ---------------- GCS ----------------
def upload_file(client, local_path, gcs_path):
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)

def read_csv(client):
    blob = client.bucket(GCS_BUCKET).blob(DATA_KEY)
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))

# ---------------- HELPERS ----------------
def get_feature_names(preprocessor):
    names = []

    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            names.extend(cols)
        else:
            try:
                ohe = trans.named_steps.get("oh", None)
                if ohe:
                    names.extend(ohe.get_feature_names_out(cols))
                else:
                    names.extend(cols)
            except:
                names.extend(cols)

    return names

class TopKEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, top_k=10):
        self.top_k = top_k

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.col = X.columns[0]
        self.top = X.iloc[:, 0].value_counts().nlargest(self.top_k).index
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return pd.DataFrame({
            self.col: X.iloc[:, 0].where(X.iloc[:, 0].isin(self.top), "other")
        })

def clean_numeric(s):
    return pd.to_numeric(
        s.astype(str).str.replace(r"[^\d.]+", "", regex=True),
        errors="coerce"
    )

def inverse_log10(x):
    return 10 ** x

# ---------------- MAIN ----------------
def run_once(dry_run=False):

    client = storage.Client(project=PROJECT_ID)
    df = read_csv(client)

    # -------- CLEAN --------
    df["zipcode"] = df["zipcode"].astype(str).str.zfill(5)
    df["make_model"] = df["make"] + "_" + df["model"]
    df["age"] = 2026 - df["year"]
    df["miles_age_ratio"] = df["mileage"] / df["age"]

    df["price_num"] = clean_numeric(df["price"])
    df["age_num"] = clean_numeric(df["age"])
    df["mileage_num"] = clean_numeric(df["mileage"])
    df["miles_age_ratio_num"] = clean_numeric(df["miles_age_ratio"])

    df = df[df["price_num"].notna()]

    if len(df) < 40:
        return {"status": "noop"}

    # -------- FEATURES --------
    cat_cols = ["make_model","color","condition","transmission","fuel","city","state"]
    num_cols = ["age_num","mileage_num","miles_age_ratio_num"]

    preprocessor = ColumnTransformer([
        ("make_model", Pipeline([
            ("topk", TopKEncoder(15)),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), ["make_model"]),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
        ("num", SimpleImputer(strategy="median"), num_cols)
    ])

    models = {
        "dt": DecisionTreeRegressor(),
        "rf": RandomForestRegressor(),
        "xgb": XGBRegressor()
    }

    X = df[cat_cols + num_cols]
    y = df["price_num"]

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2)

    base_path = f"{OUTPUT_PREFIX}/{pd.Timestamp.utcnow().strftime('%Y%m%d%H')}"

    results = {}

    for name, model in models.items():

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("clf", model)
        ])

        pipe.fit(X_tr, y_tr)
                # ---------------- FEATURE IMPORTANCE ----------------
        clf = pipe.named_steps["clf"]

        if hasattr(clf, "feature_importances_"):
            try:
                feat_names = get_feature_names(pipe.named_steps["preprocessor"])
                importances = clf.feature_importances_

                imp_df = pd.DataFrame({
                    "feature": feat_names,
                    "importance": importances
                }).sort_values("importance", ascending=False)

                # Plot top 15
                plt.figure()
                imp_df.head(15).plot.barh(x="feature", y="importance")
                plt.title(f"{name} Feature Importance")

                fi_path = f"/tmp/{name}_feature_importance.png"
                plt.savefig(fi_path, bbox_inches="tight")
                plt.close()

                if not dry_run:
                    upload_file(client, fi_path, f"{base_path}/plots/{name}_feature_importance.png")

                # ---------------- PDP (TOP 3 ONLY - SAFE ORIGINAL FEATURES) ----------------

                top_feats = imp_df["feature"].head(3).tolist()
                top_feats = [f for f in top_feats if isinstance(f, str)]

                for f in top_feats:
                    try:
                        plt.figure()

                        PartialDependenceDisplay.from_estimator(
                            pipe,
                            X_val,
                            features=[f],
                            kind="average"
                        )

                        safe_f = f.replace("/", "_").replace(" ", "_")

                        pdp_path = f"/tmp/{name}_pdp_{safe_f}.png"
                        plt.savefig(pdp_path, bbox_inches="tight")
                        plt.close()

                        if not dry_run:
                            upload_file(
                                client,
                                pdp_path,
                                f"{base_path}/plots/{name}_pdp_{safe_f}.png"
                            )

                    except Exception as e:
                        logging.warning(f"PDP failed for {name} - {f}: {e}")

            except Exception as e:
                logging.warning(f"Feature importance failed for {name}: {e}")
        # -------- SAVE MODEL --------
        local_model = f"/tmp/{name}.joblib"
        joblib.dump(pipe, local_model)

        if not dry_run:
            upload_file(client, local_model, f"{base_path}/models/{name}.joblib")

        # -------- PREDICTIONS --------
        preds = pipe.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        results[name] = mae

        out = X_val.copy()
        out["actual"] = y_val
        out["pred"] = preds

        local_csv = f"/tmp/preds_{name}.csv"
        out.to_csv(local_csv, index=False)

        if not dry_run:
            upload_file(client, local_csv, f"{base_path}/preds_{name}.csv")

    return {
        "status": "ok",
        "output_prefix": base_path,
        "mae": results
    }

# ---------------- ENTRYPOINT ----------------
def train_dt_http(request):
    try:
        result = run_once()
        return (json.dumps(result), 200)
    except Exception as e:
        logging.error(traceback.format_exc())
        return (json.dumps({"error": str(e)}), 500)