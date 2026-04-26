"""
feature_selection.py
--------------------
Part 3: Feature Selection for Titanic Dataset
- Correlation analysis
- Random Forest feature importance
- (Optional) Recursive Feature Elimination (RFE)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import os


TARGET = "Survived"


# ── 1. Correlation Analysis ───────────────────────────────────────────────────

def drop_high_correlation(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """Remove features with pairwise correlation above `threshold`."""
    num_df = df.select_dtypes(include=[np.number]).drop(columns=[TARGET], errors="ignore")
    corr_matrix = num_df.corr().abs()

    # Upper triangle only
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    if to_drop:
        print(f"[INFO] Dropping highly correlated features (>{threshold}): {to_drop}")
        df = df.drop(columns=to_drop)
    else:
        print(f"[INFO] No features exceed correlation threshold {threshold}.")

    return df


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    num_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(16, 12))
    sns.heatmap(num_df.corr(), cmap="coolwarm", center=0,
                linewidths=0.3, annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[INFO] Correlation heatmap saved → {path}")


# ── 2. Random Forest Feature Importance ──────────────────────────────────────

def random_forest_importance(df: pd.DataFrame, output_dir: str,
                              n_features: int = 20) -> list:
    X = df.drop(columns=[TARGET]).select_dtypes(include=[np.number])
    y = df[TARGET]

    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    print("\n[FEATURE IMPORTANCES – Random Forest]\n", importances.head(n_features).to_string())

    # Plot
    plt.figure(figsize=(10, 8))
    importances.head(n_features).sort_values().plot(kind="barh", color="steelblue")
    plt.xlabel("Importance Score")
    plt.title(f"Top {n_features} Features (Random Forest)")
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[INFO] Feature importance plot saved → {path}")

    # Select features above mean importance
    threshold = importances.mean()
    selected = importances[importances >= threshold].index.tolist()
    print(f"\n[INFO] Features above mean importance ({threshold:.4f}): {selected}")
    return selected


# ── 3. Recursive Feature Elimination (RFE) – Optional ────────────────────────

def rfe_selection(df: pd.DataFrame, n_features_to_select: int = 15) -> list:
    X = df.drop(columns=[TARGET]).select_dtypes(include=[np.number])
    y = df[TARGET]

    estimator = LogisticRegression(max_iter=1000, random_state=42)
    selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
    selector.fit(X, y)

    selected = X.columns[selector.support_].tolist()
    ranking = pd.Series(selector.ranking_, index=X.columns).sort_values()
    print("\n[RFE RANKING]\n", ranking.to_string())
    print(f"\n[INFO] RFE selected {n_features_to_select} features: {selected}")
    return selected


# ── Pipeline ──────────────────────────────────────────────────────────────────

def select(input_path: str, output_path: str,
           plots_dir: str = "notebooks/figures") -> pd.DataFrame:
    df = pd.read_csv(input_path)
    os.makedirs(plots_dir, exist_ok=True)

    # Step 1 – drop high-correlation features
    df = drop_high_correlation(df, threshold=0.90)

    # Step 2 – correlation heatmap
    plot_correlation_heatmap(df, plots_dir)

    # Step 3 – Random Forest importance
    selected_rf = random_forest_importance(df, plots_dir)

    # Step 4 – RFE (optional)
    selected_rfe = rfe_selection(df)

    # Final feature set: union of RF + RFE selections, always keep TARGET
    final_features = list(set(selected_rf) | set(selected_rfe) | {TARGET})
    final_features = [f for f in final_features if f in df.columns]

    df_selected = df[final_features]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_selected.to_csv(output_path, index=False)
    print(f"\n[INFO] Selected features saved → {output_path}")
    print(f"[INFO] Final feature count: {len(final_features) - 1} + target")
    return df_selected


if __name__ == "__main__":
    select(
        input_path="data/train_engineered.csv",
        output_path="data/train_selected.csv",
        plots_dir="notebooks/figures",
    )
