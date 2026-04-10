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
from matplotlib.ticker import FuncFormatter

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
        raise FileNotFoundError("gs://%s/%s not found" % (bucket, key))
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

    required = {"scraped_at", "post_id" , "price", "make", "model", "year", "mileage","transmission","color","fuel","city","state","zipcode"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: %s" % sorted(missing))
    
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
    # 2. "FIRST-SEEN" DEDUPLICATION TO PREVENT LEAKAGE
    # ---------------------------------------------------------
    # Sort ascending so the OLDEST (first) scrape of a post_id is at the top
    df = df.sort_values("scraped_at_dt_utc", ascending=True)

    # Drop duplicates, keeping ONLY the first time we ever saw this post_id
    orig_count = len(df)
    df = df.drop_duplicates(subset=["post_id"], keep="first").copy()
    
    logging.info("Deduplication: Kept %d unique cars out of %d total scrapes based on post_id." % (len(df), orig_count))

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

    logging.info("Historical Train Rows: %d", len(train_df))
    logging.info("Brand New Holdout Rows (%s): %d", today_local, len(holdout_df))

    if len(train_df) < 40:
        return {"status": "noop", "reason": "too few training rows", "train_rows": int(len(train_df))}
    
    target = "price_num"
    # Safely select categorical columns strictly available in dataframe 
    cat_cols_candidate = ["color", "condition", "transmission", "fuel", "city", "state", "zipcode"]
    cat_cols = [c for c in cat_cols_candidate if c in df.columns]

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

        cv = KFold(n_splits=3, shuffle=True, random_state=42)

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

        logging.info("[%s] Best params: %s" % (name, search.best_params_))
        
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
        local_model = "/tmp/%s.joblib" % name
        joblib.dump(best_pipe, local_model)
        logging.info("[%s] Saved model locally: %s" % (name, local_model))
        
        # ---------------- SAVE PREDICTIONS ----------------
        out = X_hold.copy()
        out["actual"] = y_hold
        out["pred"] = hold_preds
        local_csv = "/tmp/%s_preds.csv" % name
        out.to_csv(local_csv, index=False)
        logging.info("[%s] Saved preds locally: %s" % (name, local_csv))

        if not dry_run:
            gcs_model_path = "%s/models/%s.joblib" % (base_path, name)
            upload_file(client, local_model, gcs_model_path)
            logging.info("[%s] Uploaded model to GCS: %s" % (name, gcs_model_path))

        if not dry_run:
            gcs_csv_path = "%s/preds/%s_preds.csv" % (base_path, name)
            upload_file(client, local_csv, gcs_csv_path)
            logging.info("[%s] Uploaded preds to GCS: %s" % (name, gcs_csv_path))

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
                clf,
                X_hold_trans,
                y_hold,
                n_repeats=5,
                random_state=42,
                scoring="neg_mean_absolute_error",
                n_jobs=-1
            )

            imp_df = pd.DataFrame({
                "feature": feat_names,
                "importance": perm.importances_mean,
                "std": perm.importances_std
            }).sort_values("importance", ascending=False)

            # SAVE permutation importance errors
            err_df = pd.DataFrame({
                "feature": feat_names,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std
            })

            err_local = "/tmp/%s_perm_importance.csv" % name
            err_df.to_csv(err_local, index=False)

            if not dry_run:
                gcs_err_path = "%s/errors/%s_perm_importance.csv" % (base_path, name)
                upload_file(client, err_local, gcs_err_path)
                logging.info("[%s] Uploaded permutation importance errors: %s" % (name, gcs_err_path))

        except Exception as e:
            logging.warning("[%s] Permutation importance failed: %s" % (name, e))
            imp_df = pd.DataFrame({"feature": [], "importance": []})

        if imp_df.empty:
            logging.warning("[%s] No features available for plotting. Skipping." % name)
            continue
            
        agg = imp_df.groupby(
            imp_df["feature"].str.split("_").str[0]
        )["importance"].sum().sort_values(ascending=False)

        top_feats = agg.head(3).index.tolist()
        logging.info("[%s] TOP AGGREGATED FEATURES: %s" % (name, top_feats))

        # ---------------- PLOT 1: FEATURE IMPORTANCE ----------------
        plt.figure(figsize=(10, 6))
        top_15_imp = imp_df.head(15).copy()
        # Sort ascending so the largest bar ends up at the very top of the horizontal chart
        top_15_imp = top_15_imp.sort_values(by="importance", ascending=True)
            
        plt.barh(top_15_imp["feature"], top_15_imp["importance"], color="skyblue")
        plt.title("Top 15 Feature Importances - %s" % name.upper())
        plt.xlabel("Importance Score")
        plt.tight_layout()

        fi_local_path = "/tmp/%s_feature_importance.png" % name
        plt.savefig(fi_local_path)
        plt.close()

        if not dry_run:
            fi_gcs_path = "%s/plots/%s_feature_importance.png" % (base_path, name)
            upload_file(client, fi_local_path, fi_gcs_path)
            logging.info("[%s] Uploaded Feature Importance Plot to GCS: %s" % (name, fi_gcs_path))

       # ---------------- PLOT 2: PDP (Top 3 Features) ----------------
        top_encoded = imp_df.nlargest(3, "importance")["feature"].tolist()

        if len(top_encoded) == 0:
            logging.warning("[%s] No features available for PDP" % name)
            continue
        
        valid_idx = []
        valid_names = []
        for f in top_encoded:
            if f in feat_names:
                valid_idx.append(feat_names.index(f))
                valid_names.append(f)

        logging.info("[%s] Generating PDP for exact encoded features: %s" % (name, valid_names))

        # FIX 1: Generate PDP using a sample of historical TRAINING data, not the tiny holdout data
        # This ensures rare cars/features actually exist in the data we plot
        X_train_trans = pre.transform(X)
        if scipy.sparse.issparse(X_train_trans):
            X_train_trans = X_train_trans.toarray()
            
        np.random.seed(42)
        sample_size = min(2000, X_train_trans.shape[0]) # Cap at 2000 rows for fast processing
        idx_sample = np.random.choice(X_train_trans.shape[0], sample_size, replace=False)
        X_pdp_bg = X_train_trans[idx_sample]


        for idx, name_ in zip(valid_idx, valid_names):
            try:
                # Provide an explicit axes to safely render into
                fig, ax = plt.subplots(figsize=(8, 6))
                
                PartialDependenceDisplay.from_estimator(
                    inner_model,       
                    X_pdp_bg,       
                    features=[idx],
                    feature_names=feat_names, # Maps indices back to real names for axes labels
                    kind="average",
                    ax=ax
                )

                # FIX 2: Clean up the X-Axis for Binary (One-Hot Encoded) Categorical Features
                unique_vals = np.unique(X_pdp_bg[:, idx])
                if len(unique_vals) <= 2 and set(unique_vals).issubset({0.0, 1.0, 0, 1}):
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(["False (Doesn't Have)", "True (Has)"])
                    ax.set_xlim(-0.5, 1.5)

                # FIX 3: Convert the Log10 Y-Axis back into Actual Dollar Amounts without f-strings
                formatter = FuncFormatter(lambda y, pos: "$%s" % format(int(10**y), ","))
                ax.yaxis.set_major_formatter(formatter)

                safe_feat = name_.replace("/", "_").replace(" ", "_")
                path = "/tmp/%s_pdp_%s.png" % (name, safe_feat)

                plt.title("PDP for %s (%s)" % (name_, name.upper()))
                plt.tight_layout()
                plt.savefig(path, bbox_inches="tight")
                plt.close(fig)

                if not dry_run:
                    gcs_pdp_path = "%s/plots/%s_pdp_%s.png" % (base_path, name, safe_feat)
                    upload_file(client, path, gcs_pdp_path)
                    logging.info("[%s] PDP uploaded to GCS: %s" % (name, gcs_pdp_path))

            except Exception as e:
                logging.warning("[%s] PDP failed for '%s': %s" % (name, name_, e))
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