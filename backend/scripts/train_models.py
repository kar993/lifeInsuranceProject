"""Train mortality and persistency models from synthetic CSV data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.paths import ARTIFACTS_DIR, DATA_DIR, MODELS_DIR

MORTALITY_FEATURES = [
    "age",
    "bmi",
    "bmi_squared",
    "gender",
    "occupation_risk",
    "alcohol_use",
    "exercise_level",
    "smoker_status",
    "diabetes",
    "hypertension",
    "heart_disease",
    "respiratory_disease",
    "cancer_history",
]

PERSISTENCY_FEATURES = [
    "age",
    "annual_income",
    "dependents",
    "coverage_amount",
    "premium_amount",
    "policy_term",
    "premium_income_ratio",
    "coverage_income_ratio",
    "coverage_adequacy_ratio",
    "years_since_issue",
    "product_type",
    "payment_mode",
    "payment_mode_preference",
    "autopay_enabled",
    "autopay_preference",
    "payment_mode_matches_preference",
    "autopay_matches_preference",
]


def _build_classifier_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train_mortality_model() -> Path:
    df = pd.read_csv(DATA_DIR / "mortality_training_data.csv")
    df["bmi_squared"] = df["bmi"] ** 2

    x = df[MORTALITY_FEATURES]
    y = df["mortality_event"]

    model = _build_classifier_pipeline(
        numeric_features=["age", "bmi", "bmi_squared"],
        categorical_features=[
            "gender",
            "occupation_risk",
            "alcohol_use",
            "exercise_level",
            "smoker_status",
            "diabetes",
            "hypertension",
            "heart_disease",
            "respiratory_disease",
            "cancer_history",
        ],
    )
    model.fit(x, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / "mortality_prediction_model_v1.joblib"
    joblib.dump(model, output_path)
    return output_path


def train_persistency_model() -> Path:
    df = pd.read_csv(DATA_DIR / "persistency_training_data.csv")
    df["payment_mode_matches_preference"] = (
        df["payment_mode"] == df["payment_mode_preference"]
    ).astype(int)
    df["autopay_matches_preference"] = (
        df["autopay_enabled"] == df["autopay_preference"]
    ).astype(int)

    x = df[PERSISTENCY_FEATURES]
    y = df["lapse_flag"]

    model = _build_classifier_pipeline(
        numeric_features=[
            "age",
            "annual_income",
            "dependents",
            "coverage_amount",
            "premium_amount",
            "policy_term",
            "premium_income_ratio",
            "coverage_income_ratio",
            "coverage_adequacy_ratio",
            "years_since_issue",
            "autopay_enabled",
            "autopay_preference",
            "payment_mode_matches_preference",
            "autopay_matches_preference",
        ],
        categorical_features=[
            "product_type",
            "payment_mode",
            "payment_mode_preference",
        ],
    )
    model.fit(x, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / "persistency_prediction_model_v1.joblib"
    joblib.dump(model, output_path)
    return output_path


def ensure_risk_thresholds() -> None:
    mortality_thresholds = ARTIFACTS_DIR / "mortality_model" / "risk_thresholds.json"
    persistency_thresholds = (
        ARTIFACTS_DIR / "persistency_model" / "risk_thresholds.json"
    )

    if not mortality_thresholds.exists():
        mortality_thresholds.parent.mkdir(parents=True, exist_ok=True)
        mortality_thresholds.write_text(
            json.dumps(
                {
                    "low": {"min": 0.0, "max": 0.0025},
                    "medium": {"min": 0.0025, "max": 0.005},
                    "high": {"min": 0.005, "max": 0.01},
                    "severe": {"min": 0.01, "max": 1.0},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if not persistency_thresholds.exists():
        persistency_thresholds.parent.mkdir(parents=True, exist_ok=True)
        persistency_thresholds.write_text(
            json.dumps(
                {
                    "low": {"min": 0.0, "max": 0.10},
                    "medium": {"min": 0.10, "max": 0.25},
                    "high": {"min": 0.25, "max": 0.50},
                    "severe": {"min": 0.50, "max": 1.0},
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def main() -> None:
    ensure_risk_thresholds()
    mortality_path = train_mortality_model()
    persistency_path = train_persistency_model()
    print(f"Saved mortality model: {mortality_path}")
    print(f"Saved persistency model: {persistency_path}")


if __name__ == "__main__":
    main()
