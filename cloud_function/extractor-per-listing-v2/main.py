# main.py
# Purpose: Convert raw TXT -> one-line JSON records (.jsonl) in GCS.
# Compatible input layouts:
#   gs://<bucket>/<SCRAPES_PREFIX>/<RUN>/*.txt
#   gs://<bucket>/<SCRAPES_PREFIX>/<RUN>/txt/*.txt
# where <RUN> is either 20251026T170002Z or 20251026170002.
# Output:
#   gs://<bucket>/<STRUCTURED_PREFIX>/run_id=<RUN>/jsonl/<post_id>.jsonl

import os
import re
import json
import logging
import traceback
from datetime import datetime, timezone

from flask import Flask, Request, jsonify
from google.cloud import storage
import pandas as pd

# -------------------- ENV --------------------
PROJECT_ID         = os.getenv("PROJECT_ID")
BUCKET_NAME        = os.getenv("GCS_BUCKET")                        # REQUIRED
SCRAPES_PREFIX     = os.getenv("SCRAPES_PREFIX", "scrapes")         # input
STRUCTURED_PREFIX  = os.getenv("STRUCTURED_PREFIX", "structured")   # output
# -------------------- CONFIG --------------------
PROJECT_ID        = os.getenv("PROJECT_ID")
BUCKET_NAME       = os.getenv("GCS_BUCKET")                        # REQUIRED
SCRAPES_PREFIX    = os.getenv("SCRAPES_PREFIX", "scrapes")
STRUCTURED_PREFIX = os.getenv("STRUCTURED_PREFIX", "structured_v2")

# Accept BOTH run id styles:
RUN_ID_ISO_RE   = re.compile(r"^\d{8}T\d{6}Z$")  # 20251026T170002Z
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")        # 20251026170002

READ_RETRY = gax_retry.Retry(
    predicate=gax_retry.if_transient_error,
    initial=1.0, maximum=10.0, multiplier=2.0, deadline=120.0
)

storage_client = storage.Client()

# -------------------- REGEX --------------------
PRICE_RE = re.compile(r"\$\s?([\d,]+)")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
CAR_MAKES = [
    "Toyota", "Honda", "Ford", "Chevrolet", "Nissan", "BMW",
    "Mercedes", "Kia", "Hyundai", "Volkswagen", "Subaru",
    "Mazda", "Jeep", "Ram", "GMC"
]
MAKE_MODEL_RE = re.compile(r"\b(" + "|".join(CAR_MAKES) + r")\b\s+([A-Z][A-Za-z0-9]+)")

# -------------------- PARSING FUNCTION --------------------
def parse_listing(text: str) -> dict:
    d = {}

    # PRICE
    m = PRICE_RE.search(text)
    if m:
        try: d["price"] = int(m.group(1).replace(",", ""))
        except: d["price"] = None

    # YEAR
    y = YEAR_RE.search(text)
    d["year"] = int(y.group(0)) if y else None

    # MAKE / MODEL
    mm = MAKE_MODEL_RE.search(text)
    if mm:
        d["make"] = mm.group(1)
        d["model"] = mm.group(2)
    else:
        d["make"] = None
        d["model"] = None

    # MILEAGE
    mileage = None
    m1 = re.search(r"(?:mileage|odometer)\s*[:\-]?\s*([\d,]+)", text, re.I)
    if m1:
        try: mileage = int(m1.group(1).replace(",", ""))
        except: pass
    if mileage is None:
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*k\s*(?:mi|mile|miles)\b", text, re.I)
        if m2:
            try: mileage = int(float(m2.group(1)) * 1000)
            except: pass
    if mileage is None:
        m3 = re.search(r"(\d{1,3}(?:[,\d]{3})*)\s*(?:mi|mile|miles)\b", text, re.I)
        if m3:
            try: mileage = int(re.sub(r"[^\d]", "", m3.group(1)))
            except: pass
    d["mileage"] = mileage

    # COLOR
    color_m = re.search(r"^\s*paint\s*color\s*[:=\-]?\s*([^\n\r]+)", text, re.I | re.M)
    d["color"] = color_m.group(1).strip() if color_m else None

    # CONDITION
    cond_m = re.search(r"^\s*condition\s*[:=\-]?\s*([^\n\r]+)", text, re.I | re.M)
    d["condition"] = cond_m.group(1).strip() if cond_m else None

    # TRANSMISSION
    trans_m = re.search(r"^\s*transmission\s*[:=\-]?\s*([^\n\r]+)", text, re.I | re.M)
    d["transmission"] = trans_m.group(1).strip() if trans_m else None

    # FUEL
    fuel_m = re.search(r"^\s*fuel\s*[:=\-]?\s*([^\n\r]+)", text, re.I | re.M)
    d["fuel"] = fuel_m.group(1).strip() if fuel_m else None

    LOCATION_TITLE_RE = re.compile(
        r"-\s*([A-Za-z .'-]+?),\s*([A-Z]{2})(?:\s+(\d{5}))?\s*-",
        re.I
    )

    LOCATION_PAREN_RE = re.compile(
        r"\(([A-Za-z .'-]+)\)",
        re.I
    )

    LOCATION_CITY_STATE_RE = re.compile(
        r"\b([A-Za-z .'-]+?),\s*([A-Z]{2})(?:\s+(\d{5}))?",
        re.I
    )
    # LOCATION
    city = None
    state = None
    zipcode = None

    # Pattern 1: Craigslist title format
    m = LOCATION_TITLE_RE.search(text)
    if m:
        city = m.group(1).strip()
        state = m.group(2).upper()
        zipcode = m.group(3)

    # Pattern 2: Parentheses location
    if city is None:
        m = LOCATION_PAREN_RE.search(text)
        if m:
            city = m.group(1).strip().title()

    # Pattern 3: Generic City, ST
    if city is None:
        m = LOCATION_CITY_STATE_RE.search(text)
        if m:
            city = m.group(1).strip()
            state = m.group(2).upper()
            zipcode = m.group(3)

    d["city"] = city
    d["state"] = state
    d["zipcode"] = zipcode

    return d

# -------------------- HELPERS --------------------
def _download_text(blob_name: str) -> str:
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    return blob.download_as_text()

def _upload_jsonl_line(blob_name: str, record: dict):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    blob.upload_from_string(line, content_type="application/x-ndjson")

# -------------------- CLOUD FUNCTION ENTRY --------------------
def extract_http(request: Request):
    """
    HTTP-triggered Cloud Function.
    Expects optional JSON body: {"run_id": "...", "max_files": 0, "overwrite": false}
    """
    logging.getLogger().setLevel(logging.INFO)

    try:
        body = request.get_json(silent=True) or {}
    except:
        body = {}

    run_id    = body.get("run_id")
    max_files = int(body.get("max_files") or 0)
    overwrite = bool(body.get("overwrite") or False)

    if not BUCKET_NAME:
        return jsonify({"ok": False, "error": "missing GCS_BUCKET env"}), 500

    # Determine run_id
    if not run_id:
        # fallback to latest run
        runs = sorted([b.name.split("/")[-2] for b in storage_client.list_blobs(BUCKET_NAME, prefix=f"{SCRAPES_PREFIX}/") if b.name.endswith(".txt")])
        run_id = runs[-1] if runs else None
        if not run_id:
            return jsonify({"ok": False, "error": "no run_id found"}), 200

    # List .txt files for run
    txt_blobs = [b.name for b in storage_client.list_blobs(BUCKET_NAME, prefix=f"{SCRAPES_PREFIX}/{run_id}/") if b.name.endswith(".txt")]
    if max_files > 0:
        txt_blobs = txt_blobs[:max_files]

    processed = written = skipped = errors = 0
    for name in txt_blobs:
        try:
            text = _download_text(name)
            fields = parse_listing(text)

            post_id = os.path.splitext(os.path.basename(name))[0]
            record = {
                "post_id": post_id,
                "run_id": run_id,
                "source_txt": name,
                **fields,
            }

            out_key = f"{STRUCTURED_PREFIX}/run_id={run_id}/jsonl/{post_id}.jsonl"

            if not overwrite and storage_client.bucket(BUCKET_NAME).blob(out_key).exists():
                skipped += 1
            else:
                _upload_jsonl_line(out_key, record)
                written += 1

        except Exception as e:
            errors += 1
            logging.error(f"Failed {name}: {e}\n{traceback.format_exc()}")

        processed += 1

    return jsonify({
        "ok": True,
        "run_id": run_id,
        "processed_txt": processed,
        "written_jsonl": written,
        "skipped_existing": skipped,
        "errors": errors
    }), 200

# -------------------- LOCAL TEST (Optional) --------------------
if __name__ == "__main__":
    # Local folder test
    TEST_FOLDER = "data/listings"
    if os.path.exists(TEST_FOLDER):
        for f in os.listdir(TEST_FOLDER):
            if f.endswith(".txt"):
                with open(os.path.join(TEST_FOLDER, f), "r", encoding="utf-8") as fh:
                    print(parse_listing(fh.read()))