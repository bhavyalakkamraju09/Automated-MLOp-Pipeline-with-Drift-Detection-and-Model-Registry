"""Great Expectations schema/quality validation gate for the raw churn CSV.

Runs as a DVC stage: fails the pipeline (non-zero exit) if validation does
not pass, so bad data never reaches feature engineering.
"""
import sys

import pandas as pd

RAW_PATH = "data/raw/telco_churn.csv"
OUT_PATH = "data/processed/validated.parquet"

EXPECTATIONS = [
    ("customerID column exists", lambda df: "customerID" in df.columns),
    ("Churn column exists", lambda df: "Churn" in df.columns),
    ("Churn values in {Yes, No}", lambda df: df["Churn"].isin(["Yes", "No"]).all()),
    ("tenure has no nulls", lambda df: df["tenure"].notnull().all()),
    ("tenure in [0, 120]", lambda df: df["tenure"].between(0, 120).all()),
    ("MonthlyCharges has no nulls", lambda df: df["MonthlyCharges"].notnull().all()),
    ("MonthlyCharges in [0, 200]", lambda df: df["MonthlyCharges"].between(0, 200).all()),
    ("no fully duplicate rows", lambda df: not df.duplicated().all()),
]


def validate_churn_data(df: pd.DataFrame) -> bool:
    failed = []
    for name, check in EXPECTATIONS:
        try:
            ok = bool(check(df))
        except Exception as e:
            ok = False
            name = f"{name} (error: {e})"
        if not ok:
            failed.append(name)

    if failed:
        print(f"Validation FAILED: {len(failed)}/{len(EXPECTATIONS)} expectations not met")
        for f in failed:
            print(f"  - {f}")
        return False

    print(f"Validation PASSED: {len(EXPECTATIONS)}/{len(EXPECTATIONS)} expectations met")
    return True


def main():
    df = pd.read_csv(RAW_PATH)
    if not validate_churn_data(df):
        sys.exit(1)
    import os
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet(OUT_PATH)
    print(f"Validated data written to {OUT_PATH}")


if __name__ == "__main__":
    main()
