import os, io, json, logging, traceback, re
import numpy as np
import pandas as pd
from google.cloud import storage

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
import joblib

from scipy.stats import randint, uniform

# ---------------- ENV ----------------
PROJECT_ID     = os.getenv("PROJECT_ID", "")
GCS_BUCKET     = os.getenv("GCS_BUCKET", "")
DATA_KEY       = os.getenv("DATA_KEY", "structured_v2/datasets/listings_master_llm.csv")
OUTPUT_PREFIX  = os.getenv("OUTPUT_PREFIX", "preds")
TIMEZONE       = os.getenv("TIMEZONE", "America/New_York")

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")


# ---------------- GCS HELPERS ----------------
def _upload_file_to_gcs(client, bucket_name, local_path, gcs_path):
    if not isinstance(gcs_path, str):
        raise ValueError(f"gcs_path must be a string, got {type(gcs_path)}")

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)


def _read_csv_from_gcs(client, bucket, key):
    b = client.bucket(bucket)
    blob = b.blob(key)
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))


# ---------------- FEATURE ENGINEERING ----------------
class TopKEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, top_k=10):
        self.top_k = top_k

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.col = X.columns[0]
        self.top_categories_ = X.iloc[:, 0].value_counts().nlargest(self.top_k).index
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        col_values = X.iloc[:, 0]
        return pd.DataFrame({
            self.col: col_values.where(col_values.isin(self.top_categories_), "other")
        })


def _clean_numeric(s):
    s = s.astype(str).str.replace(r"[^\d.]+", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")


def inverse_log10(x):
    return 10 ** x


# ---------------- MAIN TRAINING ----------------
def run_once(dry_run=False):

    client = storage.Client(project=PROJECT_ID)
    df = _read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)

    required = {
        "scraped_at", "price", "make", "model", "year", "mileage",
        "color", "condition", "transmission", "fuel",
        "city", "state", "zipcode"
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # ---------------- TIME SPLIT ----------------
    dt = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df["date_local"] = dt.dt.date

    # ---------------- CLEAN ----------------
    df["zipcode"] = df["zipcode"].astype(str).str.zfill(5)
    df["make_model"] = df["make"] + "_" + df["model"]
    df["age"] = 2026 - df["year"]
    df["miles_age_ratio"] = df["mileage"] / df["age"]

    df["price_num"] = _clean_numeric(df["price"])
    df["year_num"] = _clean_numeric(df["year"])
    df["mileage_num"] = _clean_numeric(df["mileage"])
    df["age_num"] = _clean_numeric(df["age"])
    df["miles_age_ratio_num"] = _clean_numeric(df["miles_age_ratio"])

    unique_dates = sorted(df["date_local"].dropna().unique())
    today_local = unique_dates[-1]

    train_df = df[df["date_local"] < today_local].copy()
    holdout_df = df[df["date_local"] == today_local].copy()

    train_df = train_df[train_df["price_num"].notna()]

    if len(train_df) < 40:
        return {"status": "noop", "reason": "too few rows"}

    # ---------------- FEATURES ----------------
    target = "price_num"

    cat_cols = ["make_model", "color", "condition", "transmission", "fuel", "city", "state"]
    num_cols = ["age_num", "mileage_num", "miles_age_ratio_num"]

    make_model_pipe = Pipeline([
        ("topk", TopKEncoder(top_k=15)),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ])

    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ])

    num_pipe = SimpleImputer(strategy="median")

    preprocessor = ColumnTransformer([
        ("make_model", make_model_pipe, ["make_model"]),
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols)
    ])

    # ---------------- MODELS ----------------
    pipe_dt = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", DecisionTreeRegressor(random_state=42))
    ])

    pipe_rf = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", RandomForestRegressor(random_state=42))
    ])

    pipe_xgb = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", XGBRegressor(random_state=42))
    ])

    rf_log = TransformedTargetRegressor(
        regressor=pipe_rf,
        func=np.log10,
        inverse_func=inverse_log10
    )

    xgb_log = TransformedTargetRegressor(
        regressor=pipe_xgb,
        func=np.log10,
        inverse_func=inverse_log10
    )

    grids = [
        RandomizedSearchCV(pipe_dt, {"clf__max_depth": randint(3, 20)}, n_iter=5, cv=3, n_jobs=-1),
        RandomizedSearchCV(rf_log, {"regressor__clf__n_estimators": randint(50, 200)}, n_iter=5, cv=3, n_jobs=-1),
        RandomizedSearchCV(xgb_log, {"regressor__clf__max_depth": randint(3, 10)}, n_iter=5, cv=3, n_jobs=-1)
    ]

    labels = ["dt", "rf", "xgb"]

    X = train_df[cat_cols + num_cols]
    y = train_df[target]

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2)

    # ---------------- OUTPUT STRUCTURE ----------------
    now = pd.Timestamp.utcnow()
    base_path = f"{OUTPUT_PREFIX}/{now.strftime('%Y%m%d%H')}"
    model_path_gcs_root = f"{base_path}/models"

    os.makedirs("saved_models", exist_ok=True)

    best_err = float("inf")
    best_model = None

    models = {}
    results = {}

    # ---------------- TRAIN LOOP ----------------
    for gs, name in zip(grids, labels):

        gs.fit(X_tr, y_tr)

        model = gs.best_estimator_
        models[name] = model

        local_path = f"saved_models/{name}.joblib"
        gcs_path = f"{model_path_gcs_root}/{name}.joblib"

        joblib.dump(model, local_path)

        if not dry_run:
            _upload_file_to_gcs(client, GCS_BUCKET, local_path, gcs_path)

        preds = gs.predict(X_val)
        mae = mean_absolute_error(y_val, preds)

        results[name] = mae

        if mae < best_err:
            best_err = mae
            best_model = model

    # ---------------- SAVE BEST MODEL ----------------
    best_local = "saved_models/best.joblib"
    best_gcs = f"{model_path_gcs_root}/best.joblib"

    joblib.dump(best_model, best_local)

    if not dry_run:
        _upload_file_to_gcs(client, GCS_BUCKET, best_local, best_gcs)

    # ---------------- PREDICTIONS ----------------
    def make_preds(model):
        if holdout_df.empty:
            return None

        Xh = holdout_df[cat_cols + num_cols]
        preds = model.predict(Xh)

        out = holdout_df.copy()
        out["pred_price"] = preds
        return out

    pred_outputs = {}

    for name, model in models.items():
        df_out = make_preds(model)
        if df_out is not None:
            path = f"{base_path}/preds_{name}.csv"

            local_tmp = f"/tmp/preds_{name}.csv"
            df_out.to_csv(local_tmp, index=False)

            if not dry_run:
                _upload_file_to_gcs(client, GCS_BUCKET, local_tmp, path)

            pred_outputs[name] = path

    # ---------------- RETURN ----------------
    return {
        "status": "ok",
        "output_prefix": base_path,
        "best_mae": float(best_err),
        "mae_by_model": results,
        "models_saved": list(models.keys()),
        "pred_files": pred_outputs,
        "timezone": TIMEZONE
    }


def train_dt_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(dry_run=body.get("dry_run", False))
        return (json.dumps(result), 200, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error(traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500)