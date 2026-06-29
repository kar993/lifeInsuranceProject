"""Generate synthetic mortality-risk training data for the advisory project.

The generator uses the project mortality table and underwriting lookup tables as
its source of truth.  It emits both the model inputs described in
``projectplan_updated.md`` and audit columns that explain how the synthetic
annual mortality probability was constructed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "backend" / "data" 
DEFAULT_OUTPUT = DEFAULT_REFERENCE_DIR / "mortality_training_data.csv"

ALCOHOL_MULTIPLIERS = {"No Alcohol": 0.95, "Moderate": 1.05, "Heavy": 1.35}
EXERCISE_MULTIPLIERS = {"Low": 1.18, "Medium": 1.00, "High": 0.88}
GENDER_MULTIPLIERS = {"Female": 0.85, "Male": 1.15}
SMOKER_MULTIPLIER = 1.75

# Annual probability anchors corresponding to advisory scores 25, 50 and 75.
MORTALITY_PROBABILITY_BANDS = (0.005, 0.020, 0.050)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_reference_data(reference_dir: Path) -> tuple[pd.DataFrame, ...]:
    mortality = pd.read_csv(reference_dir / "IALM_2012-2014.csv")
    occupations = pd.read_csv(reference_dir / "01_occupation_risk_table.csv")
    bmi = pd.read_csv(reference_dir / "02_bmi_risk_table.csv")
    health = pd.read_csv(reference_dir / "03a_health_status_loading.csv")
    conditions = pd.read_csv(reference_dir / "03b_medical_condition_loading.csv")

    mortality["q_x"] = pd.to_numeric(mortality["q_x"], errors="coerce")
    mortality = mortality.dropna(subset=["q_x"]).copy()

    required_ages = set(range(18, 71))
    if not required_ages.issubset(set(mortality["Age"].astype(int))):
        raise ValueError("The mortality table must contain every age from 18 to 70.")
    if occupations["occupation"].duplicated().any():
        raise ValueError("Occupation lookup contains duplicate occupations.")
    if health["health_status"].duplicated().any():
        raise ValueError("Health-status lookup contains duplicate categories.")
    if conditions["medical_condition"].duplicated().any():
        raise ValueError("Medical-condition lookup contains duplicate conditions.")

    return mortality, occupations, bmi, health, conditions


def lookup_bmi(values: np.ndarray, bmi_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    categories = np.full(values.shape, "Unknown", dtype=object)
    multipliers = np.ones(values.shape, dtype=float)

    for row in bmi_table.itertuples(index=False):
        mask = (values >= float(row.bmi_min)) & (values < float(row.bmi_max))
        categories[mask] = row.category
        multipliers[mask] = float(row.mortality_multiplier)

    if np.any(categories == "Unknown"):
        bad = values[categories == "Unknown"][:5]
        raise ValueError(f"BMI lookup does not cover generated values: {bad}")
    return categories, multipliers


def probability_to_score(probability: np.ndarray) -> np.ndarray:
    low, medium, high = MORTALITY_PROBABILITY_BANDS
    score = np.empty_like(probability, dtype=float)

    first = probability <= low
    second = (probability > low) & (probability <= medium)
    third = (probability > medium) & (probability <= high)
    fourth = probability > high

    score[first] = probability[first] / low * 25
    score[second] = 25 + (probability[second] - low) / (medium - low) * 25
    score[third] = 50 + (probability[third] - medium) / (high - medium) * 25
    score[fourth] = 75 + (probability[fourth] - high) / (0.95 - high) * 25
    return np.clip(score, 0, 100)


def risk_category(score: np.ndarray) -> np.ndarray:
    return np.select(
        [score <= 25, score <= 50, score <= 75],
        ["Low", "Medium", "High"],
        default="Severe",
    )


def generate_mortality_data(
    records: int,
    seed: int,
    reference_dir: Path,
) -> pd.DataFrame:
    if records <= 0:
        raise ValueError("records must be positive")

    mortality, occupations, bmi_table, health_table, condition_table = (
        load_reference_data(reference_dir)
    )
    rng = np.random.default_rng(seed)

    ages = np.clip(np.rint(rng.normal(40.5, 12.0, records)), 18, 70).astype(int)
    genders = rng.choice(["Female", "Male"], records, p=[0.48, 0.52])

    height_mean = np.where(genders == "Male", 171.5, 158.5)
    height_cm = np.clip(rng.normal(height_mean, 7.0), 140, 198).round(1)
    target_bmi = np.clip(rng.normal(24.8, 4.7, records), 14.8, 44.7)
    weight_kg = (target_bmi * np.square(height_cm / 100)).round(1)
    bmi = (weight_kg / np.square(height_cm / 100)).round(1)
    bmi_categories, bmi_multipliers = lookup_bmi(bmi, bmi_table)

    occupation_rows = occupations.iloc[rng.integers(0, len(occupations), records)]
    occupation = occupation_rows["occupation"].to_numpy()
    occupation_risk = occupation_rows["risk_level"].to_numpy()
    occupation_multiplier = occupation_rows["risk_multiplier"].astype(float).to_numpy()

    smoker_probability = np.clip(
        0.12 + 0.07 * (genders == "Male") + 0.001 * (ages - 35), 0.08, 0.30
    )
    smoker_status = rng.binomial(1, smoker_probability)
    alcohol_use = rng.choice(
        ["No Alcohol", "Moderate", "Heavy"], records, p=[0.34, 0.53, 0.13]
    )
    exercise_level = rng.choice(
        ["Low", "Medium", "High"], records, p=[0.30, 0.50, 0.20]
    )

    diabetes_p = np.clip(0.025 + 0.0032 * (ages - 30) + 0.06 * (bmi >= 30), 0.01, 0.34)
    hypertension_p = np.clip(0.035 + 0.0040 * (ages - 30) + 0.07 * (bmi >= 30), 0.01, 0.42)
    heart_p = np.clip(0.008 + 0.0016 * (ages - 35) + 0.025 * smoker_status, 0.003, 0.15)
    respiratory_p = np.clip(0.035 + 0.055 * smoker_status, 0.02, 0.16)
    cancer_p = np.clip(0.006 + 0.0009 * (ages - 35), 0.003, 0.07)

    diabetes = rng.binomial(1, diabetes_p)
    hypertension = rng.binomial(1, hypertension_p)
    heart_disease = rng.binomial(1, heart_p)
    respiratory_disease = rng.binomial(1, respiratory_p)
    cancer_history = rng.binomial(1, cancer_p)

    condition_count = (
        diabetes + hypertension + heart_disease + respiratory_disease + cancer_history
    )
    health_points = (
        condition_count
        + smoker_status
        + (exercise_level == "Low").astype(int)
        + (bmi >= 35).astype(int)
    )
    health_status = np.select(
        [health_points == 0, health_points <= 1, health_points <= 3],
        ["Excellent", "Good", "Average"],
        default="Poor",
    )

    mortality_map = dict(
        zip(mortality["Age"].astype(int), mortality["q_x"].astype(float))
    )
    health_map = dict(
        zip(health_table["health_status"], health_table["mortality_multiplier"])
    )
    condition_map = dict(
        zip(condition_table["medical_condition"], condition_table["extra_loading"])
    )

    base_probability = np.array([mortality_map[age] for age in ages])
    health_multiplier = np.array([health_map[value] for value in health_status], dtype=float)
    medical_extra_loading = (
        diabetes * float(condition_map["Diabetes"])
        + hypertension * float(condition_map["Hypertension"])
        + heart_disease * float(condition_map["Heart Disease"])
        + respiratory_disease * float(condition_map["Asthma"])
        + cancer_history * float(condition_map["Cancer History"])
    )
    lifestyle_multiplier = (
        np.array([ALCOHOL_MULTIPLIERS[value] for value in alcohol_use])
        * np.array([EXERCISE_MULTIPLIERS[value] for value in exercise_level])
        * np.where(smoker_status == 1, SMOKER_MULTIPLIER, 1.0)
    )
    gender_multiplier = np.array([GENDER_MULTIPLIERS[value] for value in genders])

    mortality_probability = (
        base_probability
        * gender_multiplier
        * bmi_multipliers
        * occupation_multiplier
        * lifestyle_multiplier
        * health_multiplier
        * (1.0 + medical_extra_loading)
    )
    mortality_probability = np.clip(mortality_probability, 0.000001, 0.95)
    mortality_score = probability_to_score(mortality_probability)
    mortality_event = rng.binomial(1, mortality_probability)

    frame = pd.DataFrame(
        {
            "customer_id": [f"MORT{value:07d}" for value in range(1, records + 1)],
            "age": ages,
            "gender": genders,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": bmi,
            "bmi_category": bmi_categories,
            "occupation": occupation,
            "occupation_risk": occupation_risk,
            "smoker_status": smoker_status,
            "alcohol_use": alcohol_use,
            "exercise_level": exercise_level,
            "diabetes": diabetes,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "respiratory_disease": respiratory_disease,
            "cancer_history": cancer_history,
            "health_status": health_status,
            "base_mortality_probability": base_probability.round(6),
            "bmi_multiplier": bmi_multipliers,
            "occupation_multiplier": occupation_multiplier,
            "lifestyle_multiplier": lifestyle_multiplier.round(4),
            "health_multiplier": health_multiplier,
            "medical_extra_loading": medical_extra_loading.round(2),
            "mortality_probability": mortality_probability.round(6),
            "mortality_score": mortality_score.round(2),
            "mortality_category": risk_category(mortality_score),
            "mortality_event": mortality_event,
        }
    )
    return frame


def validate(frame: pd.DataFrame) -> None:
    if frame.isna().any().any():
        raise ValueError("Generated mortality data contains missing values.")
    if not frame["mortality_probability"].between(0, 1).all():
        raise ValueError("Mortality probabilities must be between zero and one.")
    if not frame["mortality_score"].between(0, 100).all():
        raise ValueError("Mortality scores must be between zero and 100.")
    if not set(frame["mortality_event"].unique()).issubset({0, 1}):
        raise ValueError("mortality_event must be binary.")


def main() -> None:
    args = parse_args()
    frame = generate_mortality_data(args.records, args.seed, args.reference_dir)
    validate(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print(f"Generated {len(frame):,} mortality records: {args.output}")
    print("Risk categories:")
    print(frame["mortality_category"].value_counts().sort_index().to_string())
    print(f"Observed mortality events: {int(frame['mortality_event'].sum()):,}")


if __name__ == "__main__":
    main()
