"""Generate synthetic policy-level persistency (lapse-risk) training data.

Products and rates come from the canonical project catalog.  The generated
features follow the persistency model contract in ``projectplan_updated.md``;
the probability and score columns are retained for auditability, while
``lapse_flag`` is the supervised-learning target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "backend" / "data"
DEFAULT_OUTPUT = DEFAULT_REFERENCE_DIR / "persistency_training_data.csv"

PAYMENT_MODES = np.array(["Monthly", "Quarterly", "Semi-Annual", "Annual"])
PAYMENT_MODE_PROBABILITIES = np.array([0.30, 0.15, 0.15, 0.40])
PRODUCT_LAPSE_EFFECT = {
    "Term Life": 0.00,
    "Whole Life": -0.10,
    "Endowment": 0.15,
    "Money Back": 0.10,
    "ULIP": 0.30,
    "Retirement & Pension": -0.15,
    "Child Plan": -0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_catalog(reference_dir: Path) -> pd.DataFrame:
    catalog = pd.read_csv(reference_dir / "product_catalog_sc_new.csv")
    required = {
        "product_id",
        "product_name",
        "product_type",
        "min_age",
        "max_age",
        "min_coverage",
        "max_coverage",
        "premium_paying_term_min",
        "premium_paying_term_max",
        "risk_profile",
        "base_premium_rate",
    }
    missing = required.difference(catalog.columns)
    if missing:
        raise ValueError(f"Product catalog is missing columns: {sorted(missing)}")
    if catalog["product_id"].duplicated().any():
        raise ValueError("Product catalog contains duplicate product IDs.")
    unknown_types = set(catalog["product_type"]) - set(PRODUCT_LAPSE_EFFECT)
    if unknown_types:
        raise ValueError(f"Unsupported product types: {sorted(unknown_types)}")
    return catalog


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def risk_category(score: float) -> str:
    if score <= 25:
        return "Low"
    if score <= 50:
        return "Medium"
    if score <= 75:
        return "High"
    return "Severe"


def calculate_lapse_probability(
    *,
    age: int,
    dependents: int,
    product_type: str,
    premium_income_ratio: float,
    coverage_income_ratio: float,
    policy_term: int,
    payment_mode: str,
    payment_mode_preference: str,
    autopay_enabled: int,
    years_since_issue: int,
    noise: float,
) -> float:
    """Return a bounded policy lapse probability from explainable drivers."""

    log_odds = -2.10
    log_odds += 12.0 * (min(premium_income_ratio, 0.40) - 0.05)
    log_odds += 0.035 * max(min(coverage_income_ratio, 30.0) - 12.0, 0.0)
    log_odds += 0.025 * max(policy_term - 15, 0)
    log_odds += {"Monthly": 0.20, "Quarterly": 0.10, "Semi-Annual": 0.05, "Annual": 0.0}[payment_mode]
    log_odds += 0.25 if payment_mode != payment_mode_preference else 0.0
    log_odds -= 0.65 if autopay_enabled else 0.0
    log_odds -= 0.08 * min(dependents, 4)
    log_odds += 0.30 if age < 30 else 0.0
    log_odds += PRODUCT_LAPSE_EFFECT[product_type]

    if years_since_issue <= 1:
        log_odds += 0.65
    elif years_since_issue <= 3:
        log_odds += 0.35
    elif years_since_issue >= 10:
        log_odds -= 0.20

    return float(np.clip(sigmoid(log_odds + noise), 0.01, 0.95))


def choose_product(catalog: pd.DataFrame, age: int, income: float, rng: np.random.Generator) -> pd.Series:
    eligible = catalog[(catalog["min_age"] <= age) & (catalog["max_age"] >= age)]
    if eligible.empty:
        raise ValueError(f"No catalog product is eligible for age {age}.")

    type_weight = {
        "Term Life": 2.5,
        "Whole Life": 1.1,
        "Endowment": 1.0,
        "Money Back": 0.9,
        "ULIP": 0.8 if income < 1_000_000 else 1.2,
        "Retirement & Pension": 0.8 if age < 35 else 1.4,
        "Child Plan": 1.0,
    }
    weights = eligible["product_type"].map(type_weight).to_numpy(dtype=float, copy=True)
    weights /= weights.sum()
    return eligible.iloc[int(rng.choice(len(eligible), p=weights))]


def rounded_coverage(value: float) -> int:
    return int(max(10_000, round(value / 10_000) * 10_000))


def generate_persistency_data(
    records: int,
    seed: int,
    reference_dir: Path,
) -> pd.DataFrame:
    if records <= 0:
        raise ValueError("records must be positive")

    catalog = load_catalog(reference_dir)
    rng = np.random.default_rng(seed)
    output: list[dict[str, object]] = []

    for index in range(1, records + 1):
        age = int(np.clip(round(rng.normal(40.5, 11.5)), 18, 65))
        annual_income = int(np.clip(rng.lognormal(np.log(550_000), 0.58), 150_000, 7_000_000) / 100) * 100
        dependents = int(rng.choice(np.arange(7), p=[0.19, 0.22, 0.27, 0.17, 0.09, 0.04, 0.02]))

        existing_coverage = (
            rounded_coverage(rng.uniform(0.5, 6) * annual_income)
            if rng.random() < 0.58
            else 0
        )
        outstanding_loans = (
            rounded_coverage(rng.beta(1.5, 4.0) * 8 * annual_income)
            if rng.random() < 0.82
            else 0
        )
        desired_coverage = rounded_coverage(rng.uniform(6, 18) * annual_income)
        recommended_coverage = rounded_coverage(
            max(500_000, 10 * annual_income + outstanding_loans - existing_coverage)
        )

        product = choose_product(catalog, age, annual_income, rng)
        mortality_score = float(np.clip((age - 18) * 0.75 + rng.normal(8, 8), 1, 85))
        age_loading = float(np.clip(1 + (age - 30) * 0.018, 0.78, 1.80))
        mortality_loading = 1 + mortality_score / 125
        base_rate_per_thousand = float(product["base_premium_rate"])

        # Candidate generation should normally respect affordability. A minority
        # of cases intentionally over-commit so the lapse model sees stressed
        # policies as well as suitable recommendations.
        premium_budget_ratio = 0.025 + 0.22 * float(rng.beta(2.0, 5.0))
        if rng.random() < 0.12:
            premium_budget_ratio *= float(rng.uniform(1.25, 2.0))
        affordable_coverage = (
            annual_income
            * premium_budget_ratio
            * 1_000
            / (base_rate_per_thousand * age_loading * mortality_loading)
        )
        target_coverage = max(recommended_coverage, min(desired_coverage, 1.25 * recommended_coverage))
        coverage_amount = int(
            np.clip(
                rounded_coverage(min(target_coverage, affordable_coverage)),
                int(product["min_coverage"]),
                int(product["max_coverage"]),
            )
        )

        term_min = int(product["premium_paying_term_min"])
        term_max = min(int(product["premium_paying_term_max"]), 100 - age)
        term_max = max(term_min, term_max)
        policy_term = int(rng.integers(term_min, term_max + 1))
        years_since_issue = int(rng.integers(0, min(policy_term, 20) + 1))

        payment_mode_preference = str(
            rng.choice(PAYMENT_MODES, p=PAYMENT_MODE_PROBABILITIES)
        )
        if rng.random() < 0.78:
            payment_mode = payment_mode_preference
        else:
            payment_mode = str(rng.choice(PAYMENT_MODES))

        autopay_preference = int(rng.binomial(1, 0.68))
        autopay_probability = 0.82 if autopay_preference else 0.22
        autopay_enabled = int(rng.binomial(1, autopay_probability))

        premium_amount = round(
            coverage_amount / 1_000
            * base_rate_per_thousand
            * age_loading
            * mortality_loading,
            2,
        )

        premium_income_ratio = premium_amount / annual_income
        coverage_income_ratio = coverage_amount / annual_income
        coverage_adequacy_ratio = coverage_amount / recommended_coverage

        lapse_probability = calculate_lapse_probability(
            age=age,
            dependents=dependents,
            product_type=str(product["product_type"]),
            premium_income_ratio=premium_income_ratio,
            coverage_income_ratio=coverage_income_ratio,
            policy_term=policy_term,
            payment_mode=payment_mode,
            payment_mode_preference=payment_mode_preference,
            autopay_enabled=autopay_enabled,
            years_since_issue=years_since_issue,
            noise=float(rng.normal(0, 0.30)),
        )
        persistency_score = lapse_probability * 100
        lapse_flag = int(rng.binomial(1, lapse_probability))

        output.append(
            {
                "policy_id": f"POL{index:07d}",
                "age": age,
                "annual_income": annual_income,
                "dependents": dependents,
                "existing_life_coverage": existing_coverage,
                "outstanding_loans": outstanding_loans,
                "desired_coverage": desired_coverage,
                "recommended_coverage": recommended_coverage,
                "product_id": product["product_id"],
                "product_name": product["product_name"],
                "product_type": product["product_type"],
                "product_risk_profile": product["risk_profile"],
                "coverage_amount": coverage_amount,
                "coverage_adequacy_ratio": round(coverage_adequacy_ratio, 4),
                "policy_term": policy_term,
                "payment_mode_preference": payment_mode_preference,
                "payment_mode": payment_mode,
                "autopay_preference": autopay_preference,
                "autopay_enabled": autopay_enabled,
                "years_since_issue": years_since_issue,
                "mortality_score": round(mortality_score, 2),
                "base_rate_per_thousand": base_rate_per_thousand,
                "premium_amount": premium_amount,
                "premium_income_ratio": round(premium_income_ratio, 4),
                "coverage_income_ratio": round(coverage_income_ratio, 4),
                "lapse_probability": round(lapse_probability, 4),
                "persistency_risk_score": round(persistency_score, 2),
                "persistency_risk_category": risk_category(persistency_score),
                "lapse_flag": lapse_flag,
            }
        )

    return pd.DataFrame(output)


def validate(frame: pd.DataFrame, catalog: pd.DataFrame) -> None:
    if frame.isna().any().any():
        raise ValueError("Generated persistency data contains missing values.")
    if not frame["lapse_probability"].between(0, 1).all():
        raise ValueError("Lapse probabilities must be between zero and one.")
    if not frame["persistency_risk_score"].between(0, 100).all():
        raise ValueError("Persistency risk scores must be between zero and 100.")
    if not set(frame["lapse_flag"].unique()).issubset({0, 1}):
        raise ValueError("lapse_flag must be binary.")

    limits = catalog.set_index("product_id")[["min_age", "max_age", "min_coverage", "max_coverage"]]
    checked = frame.join(limits, on="product_id")
    if not checked["age"].between(checked["min_age"], checked["max_age"]).all():
        raise ValueError("Generated policy violates product age limits.")
    if not checked["coverage_amount"].between(
        checked["min_coverage"], checked["max_coverage"]
    ).all():
        raise ValueError("Generated policy violates product coverage limits.")


def main() -> None:
    args = parse_args()
    frame = generate_persistency_data(args.records, args.seed, args.reference_dir)
    catalog = load_catalog(args.reference_dir)
    validate(frame, catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    print(f"Generated {len(frame):,} persistency records: {args.output}")
    print("Risk categories:")
    print(frame["persistency_risk_category"].value_counts().sort_index().to_string())
    print(f"Observed lapses: {int(frame['lapse_flag'].sum()):,}")


if __name__ == "__main__":
    main()
