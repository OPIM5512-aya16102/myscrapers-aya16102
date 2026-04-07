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

    # Ensure price is strictly positive for log10 transformation
    df = df[(df["price_num"].notna()) & (df["price_num"] > 0)]

    cat_cols = ["color", "condition", "transmission", "fuel", "city", "state"]
    num_cols = ["age", "mileage_num"]

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

    X = df[["make_model"] + cat_cols + num_cols]
    y = df["price_num"]

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2)
    
    base_path = pd.Timestamp.now(tz="UTC").strftime('%Y%m%d%H')
    results = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("clf", model)
        ])

        pipe.fit(X_tr, y_tr)

        # ---------------- PREDICTIONS & METRICS ----------------
        preds = pipe.predict(X_val)
        
        mae = float(mean_absolute_error(y_val, preds))
        results[name] = mae

        # ---------------- FEATURE IMPORTANCE ----------------
        clf = pipe.named_steps["clf"]

        # unwrap TransformedTargetRegressor safely
        if hasattr(clf, "regressor_"):
            inner_model = clf.regressor_
        else:
            inner_model = clf

        if hasattr(inner_model, "feature_importances_"):
            feat_names = get_feature_names(pipe.named_steps["preprocessor"])
            importances = inner_model.feature_importances_

            imp_df = pd.DataFrame({
                "feature": feat_names,
                "importance": importances
            }).sort_values("importance", ascending=False)

            agg = imp_df.groupby(
                imp_df["feature"].str.split("_").str[0]
            )["importance"].sum().sort_values(ascending=False)

            top_feats = agg.head(3).index.tolist()
            # FIX: Inserted 'name' and 'top_feats'
            logging.info(f"[{name}] TOP FEATURES: {top_feats}")

            # ---------------- PDP ----------------
            pre = pipe.named_steps["preprocessor"]
            X_val_trans = pre.transform(X_val)
            
            if scipy.sparse.issparse(X_val_trans):
                X_val_trans = X_val_trans.toarray()

            top_encoded = imp_df["feature"].head(3).tolist()

            valid_idx = []
            valid_names = []
            for f in top_encoded:
                if f in feat_names:
                    valid_idx.append(feat_names.index(f))
                    valid_names.append(f)

            # FIX: Inserted 'name' and 'valid_names'
            logging.info(f"[{name}] PDP encoded features: {valid_names}")

            for idx, name_ in zip(valid_idx, valid_names):
                try:
                    plt.figure()
                    PartialDependenceDisplay.from_estimator(
                        inner_model,       # raw unwrapped model
                        X_val_trans,       # dense encoded matrix
                        features=[idx],
                        kind="average"
                    )

                    safe_feat = name_.replace("/", "_").replace(" ", "_")
                    
                    # FIX: Interpolated actual variables
                    path = f"/tmp/{name}_pdp_{safe_feat}.png"

                    plt.savefig(path, bbox_inches="tight")
                    plt.close()

                    if not dry_run:
                        # FIX: Interpolated variables for paths
                        gcs_pdp_path = f"{OUTPUT_PREFIX}/{base_path}/plots/{name}_pdp_{safe_feat}.png"
                        upload_file(client, path, gcs_pdp_path)

                    logging.info(f"[{name}] PDP saved: {path}")

                except Exception as e:
                    # FIX: Added actual error and name vars
                    logging.warning(f"[{name}] PDP failed for '{name_}': {e}")
                    plt.close()
        else:
            # FIX: Inserted name
            logging.warning(f"[{name}] has no feature_importances_")

        # ---------------- SAVE MODEL ----------------
        # FIX: Added name
        local_model = f"/tmp/{name}.joblib"
        joblib.dump(pipe, local_model)
        logging.info(f"[{name}] Saved model locally: {local_model}")

        if not dry_run:
            # FIX: Correctly built GCS paths
            gcs_model_path = f"{OUTPUT_PREFIX}/{base_path}/models/{name}.joblib"
            upload_file(client, local_model, gcs_model_path)
            logging.info(f"[{name}] Uploaded model to GCS: {gcs_model_path}")

        # ---------------- SAVE PREDICTIONS ----------------
        out = X_val.copy()
        out["actual"] = y_val
        out["pred"] = preds

        # FIX: Added name
        local_csv = f"/tmp/{name}_preds.csv"
        out.to_csv(local_csv, index=False)
        logging.info(f"[{name}] Saved preds locally: {local_csv}")

        if not dry_run:
            # FIX: Correctly built GCS paths
            gcs_csv_path = f"{OUTPUT_PREFIX}/{base_path}/preds/{name}_preds.csv"
            upload_file(client, local_csv, gcs_csv_path)
            logging.info(f"[{name}] Uploaded preds to GCS: {gcs_csv_path}")

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