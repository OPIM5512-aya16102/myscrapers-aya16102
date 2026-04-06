import os, io, json, logging, traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor

import joblib
from scipy.stats import randint

# ---------------- ENV ----------------
PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
DATA_KEY = os.getenv("DATA_KEY", "structured_v2/datasets/listings_master_llm.csv")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "preds")
TIMEZONE = os.getenv("TIMEZONE", "America/New_York")

logging.basicConfig(level="INFO")


# ---------------- GCS HELPERS ----------------
def upload_file(client, local_path, gcs_path):
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)


def read_csv_gcs(client, key):
    blob = client.bucket(GCS_BUCKET).blob(key)
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))


# ---------------- FEATURE ENGINEERING ----------------
class TopKEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, top_k=10):
        self.top_k = top_k

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.col = X.columns[0]
        self.top_ = X.iloc[:, 0].value_counts().nlargest(self.top_k).index
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return pd.DataFrame({
            self.col: X.iloc[:, 0].where(X.iloc[:, 0].isin(self.top_), "other")
        })


def clean_num(s):
    return pd.to_numeric(s.astype(str).str.replace(r"[^\d.]+", "", regex=True), errors="coerce")


def inverse_log10(x):
    return 10 ** x


# ---------------- FEATURE NAMES ----------------
def get_feature_names(preprocessor):
    names = []

    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            names.extend(cols)

        elif name == "cat":
            oh = trans.named_steps["oh"]
            names.extend(oh.get_feature_names_out(cols))

        elif name == "make_model":
            oh = trans.named_steps["oh"]
            names.extend(oh.get_feature_names_out(["make_model"]))

    return names


# ---------------- PLOTS ----------------
def generate_plots(model, name, X_val, y_val, base_path, client):

    if isinstance(model, TransformedTargetRegressor):
        pipe = model.regressor_
    else:
        pipe = model

    pre = pipe.named_steps["preprocessor"]
    clf = pipe.named_steps["clf"]

    X_t = pre.transform(X_val)
    features = get_feature_names(pre)

    # -------- Feature Importance --------
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    else:
        perm = permutation_importance(clf, X_t, y_val, n_repeats=5)
        imp = perm.importances_mean

    df_imp = pd.DataFrame({"feature": features, "importance": imp})
    df_imp = df_imp.sort_values("importance", ascending=False)

    top3 = df_imp.head(3)["feature"].tolist()

    # Plot FI
    plt.figure()
    df_imp.head(10).plot(kind="barh", x="feature", y="importance")
    plt.gca().invert_yaxis()

    fi_local = f"/tmp/{name}_fi.png"
    plt.savefig(fi_local)
    plt.close()

    upload_file(client, fi_local, f"{base_path}/plots/{name}_fi.png")

    # -------- PDP --------
    for i, feat in enumerate(top3):
        try:
            fig, ax = plt.subplots()

            PartialDependenceDisplay.from_estimator(
                clf,
                X_t,
                [features.index(feat)],
                ax=ax
            )

            pdp_local = f"/tmp/{name}_pdp_{i}.png"
            plt.savefig(pdp_local)
            plt.close()

            upload_file(client, pdp_local, f"{base_path}/plots/{name}_pdp_{i}.png")

        except Exception as e:
            logging.warning(f"PDP failed: {e}")


# ---------------- MAIN ----------------
def run_once(dry_run=False):

    client = storage.Client(project=PROJECT_ID)
    df = read_csv_gcs(client, DATA_KEY)

    # -------- CLEAN --------
    df["date"] = pd.to_datetime(df["scraped_at"], errors="coerce").dt.date
    df["zipcode"] = df["zipcode"].astype(str).str.zfill(5)

    df["make_model"] = df["make"] + "_" + df["model"]
    df["age"] = 2026 - df["year"]
    df["ratio"] = df["mileage"] / df["age"]

    df["price_num"] = clean_num(df["price"])
    df["age"] = clean_num(df["age"])
    df["mileage"] = clean_num(df["mileage"])
    df["ratio"] = clean_num(df["ratio"])

    today = sorted(df["date"].dropna().unique())[-1]

    train = df[df["date"] < today].copy()
    hold = df[df["date"] == today].copy()

    train = train[train["price_num"].notna()]

    if len(train) < 40:
        return {"status": "noop"}

    # -------- FEATURES --------
    cat = ["make_model", "color", "condition", "transmission", "fuel", "city", "state"]
    num = ["age", "mileage", "ratio"]

    pre = ColumnTransformer([
        ("make_model", Pipeline([
            ("topk", TopKEncoder(15)),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), ["make_model"]),

        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat),

        ("num", SimpleImputer(strategy="median"), num)
    ])

    # -------- MODELS --------
    models_cfg = {
        "dt": RandomizedSearchCV(
            Pipeline([("preprocessor", pre), ("clf", DecisionTreeRegressor())]),
            {"clf__max_depth": randint(3, 20)}, n_iter=5, cv=3, n_jobs=-1
        ),

        "rf": RandomizedSearchCV(
            TransformedTargetRegressor(
                regressor=Pipeline([("preprocessor", pre), ("clf", RandomForestRegressor())]),
                func=np.log10,
                inverse_func=inverse_log10
            ),
            {"regressor__clf__n_estimators": randint(50, 200)}, n_iter=5, cv=3, n_jobs=-1
        ),

        "xgb": RandomizedSearchCV(
            TransformedTargetRegressor(
                regressor=Pipeline([("preprocessor", pre), ("clf", XGBRegressor())]),
                func=np.log10,
                inverse_func=inverse_log10
            ),
            {"regressor__clf__max_depth": randint(3, 10)}, n_iter=5, cv=3, n_jobs=-1
        )
    }

    X = train[cat + num]
    y = train["price_num"]

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2)

    now = pd.Timestamp.utcnow()
    base_path = f"{OUTPUT_PREFIX}/{now.strftime('%Y%m%d%H')}"

    results = {}
    trained = {}

    # -------- TRAIN --------
    for name, gs in models_cfg.items():

        gs.fit(X_tr, y_tr)
        model = gs.best_estimator_

        trained[name] = model

        # Save model
        local_model = f"/tmp/{name}.joblib"
        joblib.dump(model, local_model)

        if not dry_run:
            upload_file(client, local_model, f"{base_path}/models/{name}.joblib")

        # Metrics
        preds = gs.predict(X_val)
        results[name] = float(mean_absolute_error(y_val, preds))

        # 🔥 FEATURE IMPORTANCE + PDP
        if not dry_run:
            generate_plots(model, name, X_val, y_val, base_path, client)

    # -------- PREDICTIONS --------
    for name, model in trained.items():

        if hold.empty:
            continue

        preds = model.predict(hold[cat + num])
        out = hold.copy()
        out["pred_price"] = preds

        tmp = f"/tmp/preds_{name}.csv"
        out.to_csv(tmp, index=False)

        if not dry_run:
            upload_file(client, tmp, f"{base_path}/preds_{name}.csv")

    return {
        "status": "ok",
        "output_prefix": base_path,
        "mae": results
    }


# ---------------- HTTP ----------------
def train_dt_http(request):
    try:
        body = request.get_json(silent=True) or {}
        res = run_once(dry_run=body.get("dry_run", False))
        return (json.dumps(res), 200)
    except Exception as e:
        logging.error(traceback.format_exc())
        return (json.dumps({"error": str(e)}), 500)