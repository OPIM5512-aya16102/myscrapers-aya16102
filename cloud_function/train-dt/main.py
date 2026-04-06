import os, io, json, logging, traceback
import numpy as np
import pandas as pd
from google.cloud import storage

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

import joblib

# ---------------- ENV ----------------
PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
DATA_KEY = os.getenv("DATA_KEY", "structured_v2/datasets/listings_master_llm.csv")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "preds")

logging.basicConfig(level="INFO")

# ---------------- GCS ----------------
def upload_file(client, local_path, gcs_path):
    bucket = client.bucket(GCS_BUCKET)
    bucket.blob(gcs_path).upload_from_filename(local_path)

def read_csv(client):
    blob = client.bucket(GCS_BUCKET).blob(DATA_KEY)
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))

# ---------------- HELPERS ----------------
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

# ---------------- FEATURE NAMES ----------------
def get_feature_names(preprocessor):
    names = []

    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            names.extend(cols)
        else:
            try:
                ohe = trans.named_steps["oh"]
                names.extend(ohe.get_feature_names_out(cols))
            except:
                names.extend(cols)

    return names

# ---------------- MAIN ----------------
def run_once(dry_run=False):

    client = storage.Client(project=PROJECT_ID)
    df = read_csv(client)

    # -------- CLEAN --------
    df["zipcode"] = df["zipcode"].astype(str).str.zfill(5)
    df["make_model"] = df["make"] + "_" + df["model"]
    df["age"] = 2026 - df["year"]

    df["price_num"] = clean_numeric(df["price"])
    df["mileage_num"] = clean_numeric(df["mileage"])

    df = df[df["price_num"].notna()]

    cat_cols = ["make_model","color","condition","transmission","fuel","city","state"]
    num_cols = ["age","mileage_num"]

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
        "rf": TransformedTargetRegressor(
            regressor=RandomForestRegressor(),
            func=np.log10,
            inverse_func=lambda x: 10 ** x
        ),
        "xgb": TransformedTargetRegressor(
            regressor=XGBRegressor(),
            func=np.log10,
            inverse_func=lambda x: 10 ** x
        )
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
        base_model = clf.regressor_ if hasattr(clf, "regressor_") else clf

        if hasattr(base_model, "feature_importances_"):

            feat_names = get_feature_names(pipe.named_steps["preprocessor"])
            importances = base_model.feature_importances_

            imp_df = pd.DataFrame({
                "feature": feat_names,
                "importance": importances
            }).sort_values("importance", ascending=False)

            # aggregate to raw feature level
            def map_raw(f):
                for c in cat_cols + num_cols:
                    if f.startswith(c):
                        return c
                return f

            imp_df["raw"] = imp_df["feature"].apply(map_raw)

            agg = imp_df.groupby("raw")["importance"].sum().sort_values(ascending=False)

            top_feats = agg.head(3).index.tolist()

            logging.info(f"{name} TOP FEATURES: {top_feats}")

            # ---------------- PDP ----------------
            pre = pipe.named_steps["preprocessor"]
            model = pipe.named_steps["clf"]
            base_model = model.regressor_ if hasattr(model, "regressor_") else model

            # transform data
            X_val_trans = pre.transform(X_val)

            # get encoded feature names
            feat_names = get_feature_names(pre)

            # top 3 features from importance (encoded space)
            top_feats = imp_df["feature"].head(3).tolist()

            # map names → indices
            valid_idx = []
            valid_names = []

            for f in top_feats:
                if f in feat_names:
                    idx = feat_names.index(f)
                    valid_idx.append(idx)
                    valid_names.append(f)

            logging.info(f"PDP features: {valid_names}")

            for f_name, idx in zip(valid_names, valid_idx):

                try:
                    plt.figure()

                    PartialDependenceDisplay.from_estimator(
                        base_model,
                        X_val_trans,
                        features=[idx],   # IMPORTANT: index, not name
                        kind="average"
                    )

                    safe = f_name.replace("/", "_").replace(" ", "_")

                    path = f"/tmp/{name}_pdp_{safe}.png"
                    plt.savefig(path, bbox_inches="tight")
                    plt.close()

                    if not dry_run:
                        upload_file(client, path, f"{base_path}/plots/{name}_pdp_{safe}.png")

                except Exception as e:
                    logging.warning(f"PDP failed {name} {f_name}: {e}")

        # ---------------- METRICS ----------------
        preds = pipe.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        results[name] = mae

    return {"status": "ok", "mae": results}


def train_dt_http(request):
    try:
        return (json.dumps(run_once()), 200)
    except Exception as e:
        return (json.dumps({"error": str(e)}), 500)