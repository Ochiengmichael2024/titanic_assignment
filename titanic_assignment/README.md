# 🚢 Titanic Dataset — AI Assignment 2

Predictive model preparation for Titanic survival using data cleaning,
feature engineering, and feature selection.

---

## Project Structure

```
titanic_assignment/
├── data/
│   ├── train.csv                ← original dataset (add here)
│   ├── train_cleaned.csv        ← output of Part 1
│   ├── train_engineered.csv     ← output of Part 2
│   └── train_selected.csv       ← output of Part 3
├── notebooks/
│   ├── Titanic_Feature_Engineering.ipynb   ← main notebook (all parts)
│   └── figures/                ← auto-generated plots
├── scripts/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── feature_selection.py
├── README.md
└── requirements.txt
```

---

## Setup & Running

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your dataset
cp /path/to/train.csv data/train.csv

# 3. Run the notebook (recommended)
jupyter notebook notebooks/Titanic_Feature_Engineering.ipynb

# OR run scripts individually
python scripts/data_cleaning.py
python scripts/feature_engineering.py
python scripts/feature_selection.py
```

---

## Part 1 — Data Cleaning

| Column | Strategy | Reason |
|--------|----------|--------|
| `Age` | Median imputation + `AgeIsMissing` flag | ~20% missing; median robust to outliers |
| `Embarked` | Mode imputation | Only 2 rows missing |
| `Fare` | Median imputation | Only 1 missing row |
| `Cabin` | Dropped (Deck extracted first) | 77% missing — not imputable |

- **Outliers**: Capped using IQR×3 method for `Age` and `Fare`
- **Consistency**: `Sex` normalised to lowercase; duplicates removed

---

## Part 2 — Feature Engineering

| Feature | Description |
|---------|-------------|
| `FamilySize` | `SibSp + Parch + 1` |
| `IsAlone` | 1 if travelling alone |
| `Title` | Extracted from Name; grouped into Mr/Mrs/Miss/Master/Officer/Royalty/Other |
| `Deck` | First letter of Cabin; `U` if unknown |
| `AgeGroup` | Child (0–12), Teen (13–17), Adult (18–60), Senior (60+) |
| `FarePerPerson` | `Fare / FamilySize` |
| `Pclass_x_Fare` | Interaction feature |
| `Age_x_Pclass` | Interaction feature |
| `Log_Fare` | log1p transform of Fare |
| `Log_FarePerPerson` | log1p transform of FarePerPerson |

Categorical encoding: One-hot encoding for Sex, Embarked, Title, Deck, AgeGroup.  
Scaling: StandardScaler applied to continuous features.

---

## Part 3 — Feature Selection

1. **Correlation Analysis** — Features with pairwise correlation > 0.90 dropped.
2. **Random Forest Importance** — Features above mean importance score selected.
3. **RFE** — Top 15 features selected via Logistic Regression RFE (extra credit).
4. **Final set** — Union of RF + RFE selections.

### Key Findings

- `Sex` is the strongest predictor (women first rule of the era)
- `Pclass` and `Fare`/`Log_Fare` are strong socioeconomic proxies
- `Title` captures age, gender, and class in a single feature
- `FamilySize` shows a non-linear relationship: medium families (2–4) survived best
- Lone travellers had lower survival rates than those with small families
- `Deck` (from Cabin) adds cabin-location signal that `Pclass` alone misses

---

## Author

AI Assignment 2 — Titanic Dataset Analysis
