# Decision Tree: train on all data < today (local TZ); hold out today
# HTTP entrypoint: train_dt_http

import os, io, json, logging, traceback, re
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
#from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor    
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor    

# ---- ENV ----
PROJECT_ID     = os.getenv("PROJECT_ID", "")
GCS_BUCKET     = os.getenv("GCS_BUCKET", "")
DATA_KEY       = os.getenv("DATA_KEY", "structured_v2/datasets/listings_master_llm.csv")
OUTPUT_PREFIX  = os.getenv("OUTPUT_PREFIX", "preds")            # e.g., "structured/preds"
TIMEZONE       = os.getenv("TIMEZONE", "America/New_York")      # split by local day
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")

class TopKEncoder(BaseEstimator, TransformerMixin):
        def __init__(self, top_k=10):
            self.top_k = top_k

        def fit(self, X, y=None):
            # assume X is a DataFrame or 2D array
            col = X.columns[0] if hasattr(X, "columns") else 0
            self.col = col

            values = X.iloc[:, 0] if hasattr(X, "iloc") else X[:, 0]
            self.top_categories_ = values.value_counts().nlargest(self.top_k).index
            return self

        def transform(self, X):
            X = X.copy()
            col_values = X.iloc[:, 0] if hasattr(X, "iloc") else X[:, 0]

            return pd.DataFrame({
                self.col: col_values.where(col_values.isin(self.top_categories_), "other")
            })

def _read_csv_from_gcs(client: storage.Client, bucket: str, key: str) -> pd.DataFrame:
    b = client.bucket(bucket)
    blob = b.blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key} not found")
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))

def _write_csv_to_gcs(client: storage.Client, bucket: str, key: str, df: pd.DataFrame):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")

def _clean_numeric(s: pd.Series) -> pd.Series:
    # Strip $, commas, spaces; keep digits and dot
    s = s.astype(str).str.replace(r"[^\d.]+", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")

def run_once(dry_run: bool = False, max_depth: int = 12, min_samples_leaf: int = 10):
    client = storage.Client(project=PROJECT_ID)
    df = _read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)

    required = {"scraped_at", "price", "make", "model", "year", "mileage", 'color', 'condition', 'transmission',
        'fuel', 'city', 'state', 'zipcode'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --- Parse timestamps and choose local-day split ---
    dt = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df["scraped_at_dt_utc"] = dt
    try:
        df["scraped_at_local"] = df["scraped_at_dt_utc"].dt.tz_convert(TIMEZONE)
    except Exception:
        df["scraped_at_local"] = df["scraped_at_dt_utc"]
    df["date_local"] = df["scraped_at_local"].dt.date

    def clean_data(df):
        # add leading 0s
        df['zipcode'] = df['zipcode'].astype(str).str.zfill(5)
        exclude_cols = ['price', 'year', 'zipcode']

        for col in df.select_dtypes(include=['object', 'string']):
            if col not in exclude_cols:
                df[col] = df[col].astype(str).str.lower()

        df['color'] = df['color'].str.replace('gray', 'grey', case=False, regex=False)
        df['make_model'] = df['make'] + '_' + df['model']
        df['age'] = 2026 - df['year']
        df['miles_age_ratio'] = df['mileage'] / df['age']
        return(df)
    


    df = clean_data(df)
    # --- Clean numerics BEFORE counting/dropping ---
    orig_rows = len(df)
    df["price_num"]   = _clean_numeric(df["price"])
    df["year_num"]    = _clean_numeric(df["year"])
    df["mileage_num"] = _clean_numeric(df["mileage"])
    df["age_num"] = _clean_numeric(df["age"])
    df["miles_age_ratio_num"] = _clean_numeric(df["miles_age_ratio"])

    valid_price_rows = int(df["price_num"].notna().sum())
    logging.info("Rows total=%d | with valid numeric price=%d", orig_rows, valid_price_rows)

    counts = df["date_local"].value_counts().sort_index()
    logging.info("Recent date counts (local): %s", json.dumps({str(k): int(v) for k, v in counts.tail(8).items()}))

    unique_dates = sorted(d for d in df["date_local"].dropna().unique())
    if len(unique_dates) < 2:
        return {"status": "noop", "reason": "need at least two distinct dates", "dates": [str(d) for d in unique_dates]}

    today_local = unique_dates[-1]
    train_df   = df[df["date_local"] <  today_local].copy()
    holdout_df = df[df["date_local"] == today_local].copy()

    train_df = train_df[train_df["price_num"].notna()]
    dropped_for_target = int((df["date_local"] < today_local).sum()) - int(len(train_df))
    logging.info("Train rows after target clean: %d (dropped_for_target=%d)", len(train_df), dropped_for_target)
    logging.info("Holdout rows today (%s): %d", today_local, len(holdout_df))

    if len(train_df) < 40:
        return {"status": "noop", "reason": "too few training rows", "train_rows": int(len(train_df))}

    # --- Model: make, model, year_num, mileage_num -> price_num ---
    target = "price_num"
    cat_cols = ['make_model', 'color', 'condition', 'transmission',
        'fuel', 'city', 'state']
    num_cols = ["age_num", "mileage_num", "miles_age_ratio_num"]
    feats = cat_cols + num_cols
    make_model_col = ["make_model"]
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
        ("make_model", make_model_pipe, make_model_col),
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols)
    ])


    def inverse_log10(x):
        return 10 ** x

 
    # Construct some pipelines
    pipe_dt = Pipeline([('preprocessor', preprocessor),
                ('clf', DecisionTreeRegressor(random_state=42))])


    pipe_rf = Pipeline([('preprocessor', preprocessor),
                ('clf', RandomForestRegressor(random_state=42))])

    from xgboost import XGBRegressor
    pipe_xgb = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', XGBRegressor(random_state=42))])

    pipe_rf_log = TransformedTargetRegressor(
        regressor=pipe_rf,
        func=np.log10,
        inverse_func=inverse_log10
    )

    pipe_xgb_log = TransformedTargetRegressor(
        regressor=pipe_xgb,
        func=np.log10,
        inverse_func=inverse_log10
)
    grid_params_dt = [{
    'clf__max_depth': [3, 5, 10, 15, None],
    'clf__min_samples_split': [2, 5, 10, 20],
    'clf__min_samples_leaf': [1, 2, 5, 10]
}]

    grid_params_rf = [{
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [5, 10, 20, None],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 5]
    }]

    grid_params_xgb = [{
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [3, 5, 7],
        'clf__learning_rate': [0.05, 0.1],
        'clf__subsample': [0.8, 1.0],
        'clf__colsample_bytree': [0.8, 1.0]
    }]

    # Construct grid searches

    gs_dt = GridSearchCV(estimator=pipe_dt,
        param_grid=grid_params_dt,
        scoring='neg_mean_absolute_error',
        cv=10)

    gs_rf = GridSearchCV(
        estimator=pipe_rf_log,
        param_grid={
            'regressor__clf__n_estimators': [100, 200],
            'regressor__clf__max_depth': [5, 10, 20, None],
            'regressor__clf__min_samples_split': [2, 5, 10],
            'regressor__clf__min_samples_leaf': [1, 2, 5]
        },
        scoring='neg_mean_absolute_error',
        cv=5,          # 🔥 reduce from 10 → 5 for speed
        n_jobs=-1
    )

    gs_xgb = GridSearchCV(
            estimator=pipe_xgb_log,
            param_grid={
                'regressor__clf__n_estimators': [100, 200],
                'regressor__clf__max_depth': [3, 5, 7],
                'regressor__clf__learning_rate': [0.05, 0.1],
                'regressor__clf__subsample': [0.8, 1.0],
                'regressor__clf__colsample_bytree': [0.8, 1.0]
            },
            scoring='neg_mean_absolute_error',
            cv=5,
            n_jobs=-1
        )


        # List of pipelines for ease of iteration
    grids = [gs_dt,  gs_rf, gs_xgb]

        # Dictionary of pipelines and classifier types for ease of reference
    grid_dict = {0: 'Decision Tree', 1:'Random Forest', 2: 'XGBoost'}

    X_train = train_df[feats]
    y_train = train_df[target]

    print('Performing model optimizations...')

    

    # =========================================================
    # Setup
    # =========================================================
    os.makedirs("saved_models", exist_ok=True)

    best_err = np.inf
    best_clf = None
    best_gs = None

    results = []  # leaderboard

    # =========================================================
    # Loop through models
    # =========================================================
    for idx, gs in enumerate(grids):

        model_label = grid_dict[idx]
        print(f'\nEstimator: {model_label}')

        # =====================================================
        # Fit grid search
        # =====================================================
        gs.fit(X_train, y_train)

        print('Best params:', gs.best_params_)
        print('Best CV score:', gs.best_score_)

        # =====================================================
        # Safe filename
        # =====================================================
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', model_label)
        model_path = f"saved_models/{safe_name}_best.joblib"

        joblib.dump(gs.best_estimator_, model_path)
        print(f"Saved model to {model_path}")


    best_model_path = "saved_models/best_overall_model.joblib"
    joblib.dump(best_gs.best_estimator_, best_model_path)
    best_decision_tree = joblib.load("saved_models/Decision_Tree_best.joblib")
    best_model_rf = joblib.load("saved_models/Random_Forest_best.joblib")
    best_model_xgb = joblib.load("saved_models/XGBoost_best.joblib")
    best_model = joblib.load("saved_models/best_overall_model.joblib")



    X_train = train_df[feats]
    y_train = train_df[target]
    best_model.fit(X_train, y_train)

    # ---- Predict/evaluate on today's holdout (now includes actual price fields) ----
    mae_today = None
    preds_df = pd.DataFrame()
    if not holdout_df.empty:
        X_h = holdout_df[feats]
        y_hat = best_model.predict(X_h)

        cols = ["post_id", "scraped_at", "price", "make", "model", "year", "mileage", 'color', 'condition', 'transmission',
        'fuel', 'city', 'state', 'zipcode']
        preds_df = holdout_df[cols].copy()
        preds_df["actual_price"] = holdout_df["price_num"]       # cleaned numeric truth
        preds_df["pred_price"]   = np.round(y_hat, 2)

        if holdout_df["price_num"].notna().any():
            y_true = holdout_df["price_num"]
            mask = y_true.notna()
            if mask.any():
                mae_today = float(mean_absolute_error(y_true[mask], y_hat[mask]))

    # --- Output path: HOURLY folder structure ---
    now_utc = pd.Timestamp.utcnow().tz_convert("UTC")
    out_key = f"{OUTPUT_PREFIX}/{now_utc.strftime('%Y%m%d%H')}/preds.csv"

    if not dry_run and len(preds_df) > 0:
        _write_csv_to_gcs(client, GCS_BUCKET, out_key, preds_df)
        logging.info("Wrote predictions to gs://%s/%s (%d rows)", GCS_BUCKET, out_key, len(preds_df))
    else:
        logging.info("Dry run or no holdout rows; skip write. Would write to gs://%s/%s", GCS_BUCKET, out_key)

    return {
        "status": "ok",
        "today_local": str(today_local),
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "valid_price_rows": valid_price_rows,
        "mae_today": mae_today,
        "output_key": out_key,
        "dry_run": dry_run,
        "timezone": TIMEZONE,
    }

def train_dt_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(
            dry_run=bool(body.get("dry_run", False)),
            max_depth=int(body.get("max_depth", 12)),
            min_samples_leaf=int(body.get("min_samples_leaf", 10)),
        )
        code = 200 if result.get("status") == "ok" else 204
        return (json.dumps(result), code, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s", e)
        logging.error("Trace:\n%s", traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})
