"""Download raw Telco Churn data and stage it for DVC tracking.

If the Kaggle CLI/credentials aren't configured, falls back to a direct
download from a mirrored raw CSV URL so `dvc repro` never hard-fails on a
missing API key.
"""
import os
import subprocess
import sys
import urllib.request

RAW_PATH = "data/raw/telco_churn.csv"
KAGGLE_DATASET = "blastchar/telco-customer-churn"
KAGGLE_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
# Public mirror fallback (IBM sample dataset, same schema)
FALLBACK_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def ingest_via_kaggle() -> bool:
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
             "-p", "data/raw", "--unzip"],
            check=True,
        )
        downloaded = os.path.join("data/raw", KAGGLE_FILE)
        if os.path.exists(downloaded):
            os.replace(downloaded, RAW_PATH)
            return True
    except Exception as e:
        print(f"Kaggle ingest failed, falling back to direct download: {e}")
    return False


def ingest_via_fallback() -> None:
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    urllib.request.urlretrieve(FALLBACK_URL, RAW_PATH)


def main():
    os.makedirs("data/raw", exist_ok=True)
    if not ingest_via_kaggle():
        ingest_via_fallback()
    if not os.path.exists(RAW_PATH):
        print("ERROR: failed to obtain raw data via Kaggle or fallback URL.")
        sys.exit(1)
    print(f"Raw data staged at {RAW_PATH}")


if __name__ == "__main__":
    main()
