"""
data_cleaning.py
----------------
Part 1: Data Cleaning for Titanic Dataset
- Missing value handling
- Outlier handling
- Data consistency checks
- Outputs train_cleaned.csv
"""

import pandas as pd
import numpy as np
import os

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {df.shape[0]} rows × {df.shape[1]} cols from {path}")
    return df


def report_missing(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_%": pct})
    print("\n[MISSING VALUES]\n", report[report["missing_count"] > 0].to_string())


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy per column:
      Age      → median imputation + binary flag AgeIsMissing
      Embarked → mode imputation (only 2 rows missing)
      Cabin    → too many missing (~77%); dropped, Deck extracted later
      Fare     → median imputation (1 row in test set)
    """
    # Age – median + indicator
    df["AgeIsMissing"] = df["Age"].isnull().astype(int)
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Embarked – mode
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Fare – median
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Cabin – keep raw for Deck extraction then drop later
    # (handled in feature_engineering.py)

    print("[INFO] Missing values handled.")
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap extreme values using IQR method for Fare and Age.
    Values above Q3 + 3*IQR are capped (we use 3× to be conservative
    since high fares can be genuinely predictive).
    """
    for col in ["Fare", "Age"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + 3 * IQR
        lower = max(0, Q1 - 3 * IQR)
        n_outliers = ((df[col] > upper) | (df[col] < lower)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"[INFO] {col}: capped {n_outliers} outliers → [{lower:.2f}, {upper:.2f}]")
    return df


def ensure_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Normalise Sex values to lowercase
    - Strip whitespace from string columns
    - Remove duplicate rows
    """
    df["Sex"] = df["Sex"].str.lower().str.strip()
    df["Embarked"] = df["Embarked"].str.strip()

    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        print(f"[INFO] Removed {dropped} duplicate row(s).")
    else:
        print("[INFO] No duplicate rows found.")

    return df


def clean(input_path: str, output_path: str) -> pd.DataFrame:
    df = load_data(input_path)
    report_missing(df)
    df = handle_missing(df)
    df = handle_outliers(df)
    df = ensure_consistency(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[INFO] Cleaned data saved → {output_path}")
    return df


if __name__ == "__main__":
    clean(
        input_path="data/train.csv",
        output_path="data/train_cleaned.csv",
    )
