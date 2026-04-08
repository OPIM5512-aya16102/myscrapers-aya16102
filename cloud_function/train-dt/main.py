import os, io, json, logging, traceback
import numpy as np
import pandas as pd
import scipy.sparse
from google.cloud import storage
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.model_selection import TimeSeriesSplit

# ---------------- ENV ----------------
PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
DATA_KEY = os.getenv("DATA_KEY", "structured_v2/datasets/listings_master_llm.csv")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "preds")

logging.basicConfig(level=logging.INFO)

# ---------------- GCS ----------------
def upload_file(client, local_path, gcs_path):
    bucket = client.bucket(GCS_BUCKET)
    bucket.blob(gcs_path).upload_from_filename(local_path)

def read_csv(client):
    blob = client.bucket(GCS_BUCKET).blob(DATA_KEY)
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))

# ---------------- HELPERS ----------------
def log10_transform(x):
    return np.log10(x)

def inverse_log10(x):
    return 10 ** x

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
        
    def get_feature_names_out(self, input_features=None):
        return np.array([self.col]) if hasattr(self, 'col') else np.array(input_features or ["col"])

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
                # Handle pipeline nested OHE safely
                ohe = trans.named_steps["oh"] if isinstance(trans, Pipeline) else trans
                names.extend(ohe.get_feature_names_out(cols))
            except Exception:
                names.extend(cols)
    return names
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

def run_backtest(dry_run: bool = False):
    client = storage.Client(project=PROJECT_ID)
    df = _read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)

    TIMEZONE = "America/New_York"

    # ---------------- CLEAN ----------------
    df["scraped_at"] = pd.to_datetime(df["scraped_at"], utc=True, errors="coerce")
    df["date_local"] = df["scraped_at"].dt.tz_convert(TIMEZONE).dt.date

    df["zipcode"] = df["zipcode"].astype(str).str.zfill(5)
    df["make_model"] = df["make"] + "_" + df["model"]

    current_year = pd.Timestamp.now(tz=TIMEZONE).year
    df["age"] = current_year - df["year"]

    df["price_num"] = clean_numeric(df["price"])
    df["mileage_num"] = clean_numeric(df["mileage"])

    df = df[(df["price_num"].notna()) & (df["price_num"] > 0)]

    # -------- DEDUP (CRITICAL) --------
    df = df.sort_values("scraped_at")
    df = df.drop_duplicates("post_id", keep="first")

    # ---------------- FEATURES ----------------
    target = "price_num"
    cat_cols = ["color", "condition", "transmission", "fuel", "city", "state", "zipcode"]
    num_cols = ["age", "mileage_num"]
    feats = cat_cols + num_cols + ["make_model"]

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
            func=log10_transform,
            inverse_func=inverse_log10
        ),
        "xgb": TransformedTargetRegressor(
            regressor=XGBRegressor(),
            func=log10_transform,
            inverse_func=inverse_log10
        )
    }

    param_grids = {
        "dt": {
            "clf__max_depth": [5, 10, 15, None],
            "clf__min_samples_leaf": [1, 5, 10]
        },
        "rf": {
            "clf__regressor__n_estimators": [100, 200],
            "clf__regressor__max_depth": [10, None],
            "clf__regressor__min_samples_leaf": [1, 5]
        },
        "xgb": {
            "clf__regressor__n_estimators": [100, 200],
            "clf__regressor__max_depth": [3, 6],
            "clf__regressor__learning_rate": [0.05, 0.1]
        }
    }

    # ---------------- ROLLING BACKTEST ----------------
    all_dates = sorted(df["date_local"].dropna().unique())

    start_date = pd.to_datetime("2026-04-01").date()
    backtest_dates = [d for d in all_dates if d >= start_date]

    results = []

    for current_day in backtest_dates:
        logging.info(f"=== Backtesting for day: {current_day} ===")

        train_df = df[df["date_local"] < current_day].copy()

        # HOLDOUT = LAST 2 DAYS
        holdout_df = df[
            (df["date_local"] >= current_day) &
            (df["date_local"] < current_day + pd.Timedelta(days=2))
        ].copy()

        if len(train_df) < 50 or len(holdout_df) < 10:
            continue

        X_train = train_df[feats]
        y_train = train_df[target]

        X_hold = holdout_df[feats]
        y_hold = holdout_df[target]

        for name, model in models.items():
            pipe = Pipeline([
                ("preprocessor", clone(preprocessor)),
                ("clf", model)
            ])

            cv = KFold(n_splits=3, shuffle=True, random_state=42)

            search = RandomizedSearchCV(
                pipe,
                param_distributions=param_grids[name],
                n_iter=8,
                scoring="neg_mean_absolute_error",
                cv=cv,
                n_jobs=-1,
                random_state=42
            )

            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            preds = best_model.predict(X_hold)

            # -------- METRICS --------
            mae = mean_absolute_error(y_hold, preds)
            rmse = np.sqrt(np.mean((y_hold - preds) ** 2))
            r2 = 1 - np.sum((y_hold - preds)**2) / np.sum((y_hold - y_hold.mean())**2)
            bias = np.mean(preds - y_hold)

            mask = y_hold != 0
            mape = np.mean(np.abs((y_hold[mask] - preds[mask]) / y_hold[mask])) * 100 if mask.any() else np.nan

            results.append({
                "date": current_day,
                "model": name,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "r2": r2,
                "bias": bias
            })

            logging.info(f"[{name}] {current_day} | MAE={mae:.2f}")

    df_results = pd.DataFrame(results)

    # ---------------- SAVE ----------------
    local_path = "/tmp/backtest_metrics.csv"
    df_results.to_csv(local_path, index=False)

    if not dry_run:
        upload_file(client, local_path, f"{OUTPUT_PREFIX}/backtest_metrics.csv")

    return df_results

def run_once(dry_run: bool = False, max_depth: int = 12, min_samples_leaf: int = 10):
    client = storage.Client(project=PROJECT_ID)
    df = _read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)

    required = {"scraped_at", "post_id" , "price", "make", "model", "year", "mileage","transmission","color","fuel","city","state","zipcode"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    TIMEZONE = "America/New_York"
    # --- Parse timestamps and choose local-day split ---
    dt = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df["scraped_at_dt_utc"] = dt
    try:
        df["scraped_at_local"] = df["scraped_at_dt_utc"].dt.tz_convert(TIMEZONE)
    except Exception:
        df["scraped_at_local"] = df["scraped_at_dt_utc"]
    df["date_local"] = df["scraped_at_local"].dt.date

    # -------- CLEAN --------
    current_year = pd.Timestamp.now(tz="America/New_York").year
    df["zipcode"] = df["zipcode"].astype(str).str.zfill(5)
    df["make_model"] = df["make"] + "_" + df["model"]
    df["age"] = current_year - df["year"]

    df["price_num"] = clean_numeric(df["price"])
    df["mileage_num"] = clean_numeric(df["mileage"])
    # Ensure price is strictly positive for log10 transformation
    df = df[(df["price_num"].notna()) & (df["price_num"] > 0)]

    # ---------------------------------------------------------
    # Sort ascending so the OLDEST (first) scrape of a post_id is at the top
    df = df.sort_values("scraped_at_dt_utc", ascending=True)

    orig_count = len(df)
    
    logging.info(f"Deduplication: Kept {len(df)} unique cars out of {orig_count} total scrapes based on post_id.")


    # ---------------------------------------------------------
    # 3. SPLIT DATA (TRAIN ON HISTORY, TEST ON BRAND NEW)
    # ---------------------------------------------------------
    unique_dates = sorted(d for d in df["date_local"].dropna().unique())
    if len(unique_dates) < 2:
        return {"status": "noop", "reason": "need at least two distinct dates", "dates": [str(d) for d in unique_dates]}

    today_local = unique_dates[-1]
    
    # Train on all unique post_ids that hit the market BEFORE today
    train_df   = df[df["date_local"] < today_local].copy()
    
    # Test ONLY on brand new post_ids that hit the market TODAY
    holdout_df = df[df["date_local"] == today_local].copy()


    unique_dates = sorted(d for d in df["date_local"].dropna().unique())

    if len(unique_dates) < 3:
        return {
            "status": "noop",
            "reason": "need at least 3 distinct dates for 2-day holdout",
            "dates": [str(d) for d in unique_dates]
        }

    # ✅ Last 2 days = holdout
    holdout_dates = unique_dates[-2:]

    # ✅ Everything before = training
    train_df   = df[df["date_local"] < holdout_dates[0]].copy()
    holdout_df = df[df["date_local"].isin(holdout_dates)].copy()

    logging.info(f"Train date range: < {holdout_dates[0]}")
    logging.info(f"Holdout dates: {holdout_dates}")
    logging.info(f"Train rows: {len(train_df)}")
    logging.info(f"Holdout rows: {len(holdout_df)}")

    logging.info("Historical Train Rows: %d", len(train_df))
    logging.info("Brand New Holdout Rows (%s): %d", today_local, len(holdout_df))

    if len(train_df) < 40:
        return {"status": "noop", "reason": "too few training rows", "train_rows": int(len(train_df))}
    

    target = "price_num"
    cat_cols = ["color", "condition", "transmission", "fuel", "city", "state", "zipcode"]
    num_cols = ["age", "mileage_num"]
    feats = cat_cols + num_cols + ["make_model"]
    param_grids = {
        "dt": {
            "clf__max_depth": [5, 10, 15, 20, None],
            "clf__min_samples_leaf": [1, 5, 10, 20]
        },
        "rf": {
            "clf__regressor__n_estimators": [100, 200],
            "clf__regressor__max_depth": [10, 20, None],
            "clf__regressor__min_samples_leaf": [1, 5, 10]
        },
        "xgb": {
            "clf__regressor__n_estimators": [100, 200],
            "clf__regressor__max_depth": [3, 6, 10],
            "clf__regressor__learning_rate": [0.05, 0.1],
            "clf__regressor__subsample": [0.8, 1.0]
        }
    }

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
            func=log10_transform,
            inverse_func=inverse_log10
        ),
        "xgb": TransformedTargetRegressor(
            regressor=XGBRegressor(),
            func=log10_transform,
            inverse_func=inverse_log10
        )
    }

    train_df = train_df.sort_values("scraped_at_dt_utc")
    X = train_df[feats]
    y = train_df[target]

    X_hold = holdout_df[feats]
    y_hold = holdout_df[target]
    
    base_path = pd.Timestamp.now(tz="UTC").strftime('%Y%m%d%H')
    results = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("clf", model)
        ])

        cv = TimeSeriesSplit(n_splits=3)

        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grids[name],
            n_iter=10,
            scoring="neg_mean_absolute_error",
            cv=cv,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )

        search.fit(X, y)  

        best_pipe = search.best_estimator_

        logging.info(f"[{name}] Best params: {search.best_params_}")
        
        # Cross-validation MAE (replaces val_mae)
        cv_mae = -search.best_score_

        # 2. HOLDOUT predictions (REAL unseen evaluation)
        hold_preds = best_pipe.predict(X_hold)
        hold_mae = float(mean_absolute_error(y_hold, hold_preds))

        results[name] = {
            "cv_mae": float(cv_mae),
            "holdout_mae": hold_mae
        }


        # ---------------- SAVE MODEL ----------------
        # FIX: Added name
        local_model = f"/tmp/{name}.joblib"
        joblib.dump(best_pipe, local_model)
        logging.info(f"[{name}] Saved model locally: {local_model}")
        # ---------------- SAVE PREDICTIONS ----------------
        out = X_hold.copy()
        out["actual"] = y_hold
        out["pred"] = hold_preds
        local_csv = f"/tmp/{name}_preds.csv"
        out.to_csv(local_csv, index=False)
        logging.info(f"[{name}] Saved preds locally: {local_csv}")

        if not dry_run:
            # FIX: Correctly built GCS paths
            gcs_model_path = f"{OUTPUT_PREFIX}/{base_path}/models/{name}.joblib"
            upload_file(client, local_model, gcs_model_path)
            logging.info(f"[{name}] Uploaded model to GCS: {gcs_model_path}")

        if not dry_run:
            # FIX: Correctly built GCS paths
            gcs_csv_path = f"{OUTPUT_PREFIX}/{base_path}/preds/{name}_preds.csv"
            upload_file(client, local_csv, gcs_csv_path)
            logging.info(f"[{name}] Uploaded preds to GCS: {gcs_csv_path}")

        

        
        # ---------------- PERMUTATION IMPORTANCE ----------------
        clf = best_pipe.named_steps["clf"]

        # unwrap TransformedTargetRegressor safely
        if hasattr(clf, "regressor_"):
            inner_model = clf.regressor_
        else:
            inner_model = clf

        
        try:
            pre = best_pipe.named_steps["preprocessor"]
            X_hold_trans = pre.transform(X_hold)

            if scipy.sparse.issparse(X_hold_trans):
                X_hold_trans = X_hold_trans.toarray()

            feat_names = get_feature_names(pre)

            perm = permutation_importance(
                best_pipe,
                X_hold,
                y_hold,
                n_repeats=5,
                random_state=42,
                scoring="neg_mean_absolute_error",
                n_jobs=-1
            )
            feat_names = X_hold.columns

            imp_df = pd.DataFrame({
                "feature": feat_names,
                "importance": perm.importances_mean,
                "std": perm.importances_std
            }).sort_values("importance", ascending=False)

            # ✅ SAVE permutation importance errors
            err_df = pd.DataFrame({
                "feature": feat_names,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std
            })

            err_local = f"/tmp/{name}_perm_importance.csv"
            err_df.to_csv(err_local, index=False)

            if not dry_run:
                gcs_err_path = f"{OUTPUT_PREFIX}/{base_path}/errors/{name}_perm_importance.csv"
                upload_file(client, err_local, gcs_err_path)
                logging.info(f"[{name}] Uploaded permutation importance errors: {gcs_err_path}")

        except Exception as e:
            logging.warning(f"[{name}] Permutation importance failed: {e}")
            imp_df = pd.DataFrame({"feature": [], "importance": []})

            # ✅ FIX: Moved all of this logic OUT of the except block!
        if imp_df.empty:
            logging.warning(f"[{name}] No features available for plotting. Skipping.")
            continue
            
        agg = imp_df.groupby(
            imp_df["feature"].str.split("_").str[0]
        )["importance"].sum().sort_values(ascending=False)

        top_feats = agg.head(3).index.tolist()
        # FIX: Inserted 'name' and 'top_feats'
        logging.info(f"[{name}] TOP AGGREGATED FEATURES: {top_feats}")

        

        # ---------------- PLOT 1: FEATURE IMPORTANCE ----------------
        plt.figure(figsize=(10, 6))
        top_15_imp = imp_df.head(15).copy()
        # Sort ascending so the largest bar ends up at the very top of the horizontal chart
        top_15_imp = top_15_imp.sort_values(by="importance", ascending=True)
            
        plt.barh(top_15_imp["feature"], top_15_imp["importance"], color="skyblue")
        plt.title(f"Top 15 Feature Importances - {name.upper()}")
        plt.xlabel("Importance Score")
        plt.tight_layout()

        fi_local_path = f"/tmp/{name}_feature_importance.png"
        plt.savefig(fi_local_path)
        plt.close()

        if not dry_run:
            fi_gcs_path = f"{OUTPUT_PREFIX}/{base_path}/plots/{name}_feature_importance.png"
            upload_file(client, fi_local_path, fi_gcs_path)
            logging.info(f"[{name}] Uploaded Feature Importance Plot to GCS: {fi_gcs_path}")

       # ---------------- PLOT 2: PDP (Top 3 Features) ----------------
        
        top_encoded = imp_df.nlargest(3, "importance")["feature"].tolist()

        if len(top_encoded) == 0:
            logging.warning(f"[{name}] No features available for PDP")
            continue
        
        valid_idx = []
        valid_names = []
        for f in top_encoded:
            if f in feat_names:
                valid_idx.append(feat_names.index(f))
                valid_names.append(f)

        logging.info(f"[{name}] Generating PDP for exact encoded features: {valid_names}")


        for feature in top_feats:
            try:
                # Provide an explicit axes to safely render into
                fig, ax = plt.subplots(figsize=(8, 6))
                
                PartialDependenceDisplay.from_estimator(
                    best_pipe,       
                    X,       
                    features=[feature],
                    kind="average",
                    ax=ax
                )

                safe_feat = name_.replace("/", "_").replace(" ", "_")
                path = f"/tmp/{name}_pdp_{safe_feat}.png"

                plt.title(f"PDP for {safe_feat} ({name.upper()})")
                plt.tight_layout()
                plt.savefig(path, bbox_inches="tight")
                plt.close(fig)

                if not dry_run:
                    gcs_pdp_path = f"{OUTPUT_PREFIX}/{base_path}/plots/{name}_pdp_{safe_feat}.png"
                    upload_file(client, path, gcs_pdp_path)
                    logging.info(f"[{safe_feat}] PDP uploaded to GCS: {gcs_pdp_path}")

            except Exception as e:
                logging.warning(f"[{name}] PDP failed for '{name_}': {e}")
                plt.close('all')
    
    return {"status": "ok", "mae": results}
            


def train_dt_http(request):
    try:
        return (json.dumps(run_once()), 200, {'Content-Type': 'application/json'})
    except Exception as e:
        return (
            json.dumps({"error": str(e), "trace": traceback.format_exc()}), 
            500, 
            {'Content-Type': 'application/json'}
        )