import pandas as pd
import xgboost as xgb
import joblib

from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ---------------------------
# Config
# ---------------------------
CSV_PATH = "bodyPerformance.csv"

RENAME_MAP = {
    "body fat_%": "body_fat",
    "gripForce": "grip_force",
    "sit and bend forward_cm": "sit_bend_forward",
    "sit-ups counts": "sit_ups",
    "broad jump_cm": "broad_jump",
    "class": "target",
    "BMI": "bmi"
}

FEATURE_COLS = [
    "age",
    "gender",
    "height_cm",
    "weight_kg",
    "body_fat",
    "diastolic",
    "systolic",
    "grip_force",
    "sit_bend_forward",
    "sit_ups",
    "broad_jump",
    "bmi",
    "strength_ratio"
]

TARGET_COL = "target"

TARGET_MAP = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3
}

BEST_PARAMS = {
    "colsample_bytree": 0.8,
    "gamma": 0,
    "learning_rate": 0.05,
    "max_depth": 7,
    "min_child_weight": 5,
    "n_estimators": 300,
    "subsample": 1.0
}


# ---------------------------
# Load Data
# ---------------------------
def load_data():
    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns=RENAME_MAP)
    return df


# ---------------------------
# Preprocess Data
# ---------------------------
def preprocess_data(df: pd.DataFrame):
    df = df.copy()
    df = df.rename(columns=RENAME_MAP)

    # Add BMI if not موجود
    if "bmi" not in df.columns:
        df["bmi"] = df["weight_kg"] / ((df["height_cm"] / 100) ** 2)

    # Add engineered feature
    df["strength_ratio"] = df["grip_force"] / df["weight_kg"]

    # Encode gender
    gender_series = df["gender"].astype(str).str.strip().str.upper()
    gender_map = {
        "F": 0,
        "M": 1,
        "0": 0,
        "1": 1
    }
    df["gender"] = gender_series.map(gender_map)

    # Encode target
    target_series = df[TARGET_COL].astype(str).str.strip().str.upper()
    y = target_series.map(TARGET_MAP)

    X = df[FEATURE_COLS].copy()

    # Validation
    if X.isnull().any().any():
        null_cols = X.columns[X.isnull().any()].tolist()
        raise ValueError(f"Some feature values could not be encoded correctly. Problem columns: {null_cols}")

    if y.isnull().any():
        bad_vals = sorted(df[TARGET_COL].dropna().astype(str).unique().tolist())
        raise ValueError(f"Unexpected target values found: {bad_vals}")

    return X, y


# ---------------------------
# Train + Evaluate + Save
# ---------------------------
def main():
    print("Loading data...")
    df = load_data()
    X, y = preprocess_data(df)

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
        **BEST_PARAMS
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 5-Fold CV Accuracy
    print("Running 5-Fold CV Accuracy...")
    cv_accuracy_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    # 5-Fold CV Macro F1
    print("Running 5-Fold CV Macro F1...")
    macro_f1_scorer = make_scorer(f1_score, average="macro")
    cv_f1_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring=macro_f1_scorer,
        n_jobs=-1
    )

    print("\n===== RESULTS =====")
    print("Fold Accuracy Scores:", cv_accuracy_scores)
    print("Mean CV Accuracy:", round(cv_accuracy_scores.mean(), 4))
    print("CV Accuracy Std:", round(cv_accuracy_scores.std(), 4))
    print("Mean CV Macro F1:", round(cv_f1_scores.mean(), 4))
    print("CV Macro F1 Std:", round(cv_f1_scores.std(), 4))

    # Train final model on all data
    print("\nTraining final model on full dataset...")
    model.fit(X, y)

    # Feature importance
    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # Save everything in one file
    bundle = {
        "model": model,
        "cv_accuracy_mean": float(cv_accuracy_scores.mean()),
        "cv_accuracy_std": float(cv_accuracy_scores.std()),
        "cv_f1_mean": float(cv_f1_scores.mean()),
        "cv_f1_std": float(cv_f1_scores.std()),
        "cv_accuracy_scores": cv_accuracy_scores.tolist(),
        "importance_df": importance_df
    }

    joblib.dump(bundle, "fitness_model_bundle.pkl")
    print("\n✅ Model saved as fitness_model_bundle.pkl")


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    main()