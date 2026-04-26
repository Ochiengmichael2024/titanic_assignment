"""
feature_engineering.py
-----------------------
Part 2: Feature Engineering for Titanic Dataset
- Derived features
- Categorical encoding
- Interaction features
- Feature transformations (log, scaling)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import re
import os


# ── 1. Derived Features ──────────────────────────────────────────────────────

def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def extract_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extract title from Name, then group rare titles."""
    df["Title"] = df["Name"].str.extract(r",\s*([^.]+)\.")

    # Normalise common titles
    title_map = {
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Lady": "Royalty", "Countess": "Royalty", "Capt": "Officer",
        "Col": "Officer", "Don": "Royalty", "Dr": "Officer",
        "Major": "Officer", "Rev": "Officer", "Sir": "Royalty",
        "Jonkheer": "Royalty", "Dona": "Royalty",
    }
    df["Title"] = df["Title"].replace(title_map)

    # Collapse anything rare into "Other"
    common = {"Mr", "Mrs", "Miss", "Master", "Officer", "Royalty"}
    df["Title"] = df["Title"].apply(lambda t: t if t in common else "Other")
    return df


def extract_deck(df: pd.DataFrame) -> pd.DataFrame:
    """Extract deck letter from Cabin; fill missing as 'U' (Unknown)."""
    df["Deck"] = df["Cabin"].apply(
        lambda c: re.findall(r"[A-Z]", str(c))[0] if pd.notna(c) and str(c) != "nan" else "U"
    )
    return df


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 12, 17, 60, 120]
    labels = ["Child", "Teen", "Adult", "Senior"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True)
    return df


def add_fare_per_person(df: pd.DataFrame) -> pd.DataFrame:
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
    return df


# ── 2. Categorical Encoding ───────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode nominal: Sex, Embarked, Title, Deck, AgeGroup
    Ordinal: Pclass already numeric (1=high, 3=low) — left as-is.
    """
    ohe_cols = ["Sex", "Embarked", "Title", "Deck", "AgeGroup"]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)

    # Convert bool columns produced by get_dummies to int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


# ── 3. Interaction Features ───────────────────────────────────────────────────

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Pclass_x_Fare"] = df["Pclass"] * df["Fare"]
    # Age × Pclass: proxy for "young & rich" vs "old & poor" combinations
    df["Age_x_Pclass"] = df["Age"] * df["Pclass"]
    return df


# ── 4. Feature Transformations ────────────────────────────────────────────────

def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Log1p-transform right-skewed features."""
    for col in ["Fare", "FarePerPerson"]:
        df[f"Log_{col}"] = np.log1p(df[col])
    return df


def scale_features(df: pd.DataFrame, fit: bool = True,
                   scaler: StandardScaler = None):
    """
    Standardise continuous features.
    Returns (df, fitted_scaler) so the same scaler can be applied to test data.
    """
    scale_cols = ["Age", "Fare", "FamilySize", "FarePerPerson",
                  "Log_Fare", "Log_FarePerPerson"]
    scale_cols = [c for c in scale_cols if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
    else:
        df[scale_cols] = scaler.transform(df[scale_cols])

    return df, scaler


# ── Pipeline ──────────────────────────────────────────────────────────────────

def engineer(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    df = add_family_features(df)
    df = extract_title(df)
    df = extract_deck(df)
    df = add_age_group(df)
    df = add_fare_per_person(df)
    df = add_interaction_features(df)
    df = log_transform(df)
    df = encode_categoricals(df)

    # Drop columns no longer needed after encoding / extraction
    drop_cols = ["Name", "Ticket", "Cabin", "PassengerId"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df, _ = scale_features(df, fit=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Feature-engineered data saved → {output_path}")
    print(f"[INFO] Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    engineer(
        input_path="data/train_cleaned.csv",
        output_path="data/train_engineered.csv",
    )
