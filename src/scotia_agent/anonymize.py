"""Generate a Scotia-shaped anonymized sample CSV from a real export.

The goal is to preserve:
- the original Scotia CSV structure expected by `parser.py`
- broad spending patterns and grouped-category behavior
- recurring-payment shape (for trends/subscriptions)

While removing:
- real merchant names
- exact locations
- exact dates
- exact amounts

Usage:
    uv run python -m scotia_agent.anonymize \
      --input data/raw/Scene_Visa_card_9024_041126.csv \
      --output data/sample_anonymized.csv
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd

from .enrich import prepare_dataframe
from .parser import load_transactions

SCOTIA_COLUMNS = [
    "Filter",
    "Date",
    "Description",
    "Sub-description",
    "Status",
    "Type of Transaction",
    "Amount",
]

CATEGORY_BASE_MERCHANT: dict[str, str] = {
    "alcohol": "LCBO BOTTLE SHOP",
    "bank_fees": "SCOTIA CREDIT CARD PROTECT",
    "bubble_tea": "COCO BUBBLE TEA",
    "car_rental": "COMMUNAUTO SHARE",
    "coffee": "TIM HORTONS CAFE",
    "dessert": "BIG SCOOPS DESSERT",
    "education": "QUEENS LEARNING PORTAL",
    "entertainment": "TICKETMASTER EVENTS",
    "fast_food": "MCDONALDS QUICK BITE",
    "fitness": "GRAVITY CLIMBING GYM",
    "food_delivery": "UBER EATS DELIVERY",
    "fuel": "PETRO-CANADA FUEL",
    "gaming": "APPLE.COM/BILL GAMING",
    "government_fees": "IMMIGRATION CANADA ONLINE",
    "groceries": "JAMES STREET MARKET",
    "insurance": "MANULIFE TRAVEL INSURANCE",
    "parking": "PRECISE PARKLINK",
    "payment": "PAYMENT FROM CHEQUING",
    "personal_care": "FAMILY HAIR CUT",
    "pharmacy": "SHOPPERS DRUG MART",
    "restaurant": "MENYA KYU KITCHEN",
    "rideshare": "UBER TRIP",
    "shopping_online": "AMAZON.CA MARKETPLACE",
    "shopping_retail": "DOLLARAMA STORE",
    "subscription_ai": "OPENAI SUBSCRIPTION",
    "subscription_cloud": "GOOGLE ONE STORAGE",
    "subscription_media": "SPOTIFY SUBSCRIPTION",
    "subscription_pro": "LINKEDIN PREMIUM",
    "subscription_shopping_online": "AMZN MKTP SUBSCRIPTION",
    "telecom": "ROGERS WIRELESS",
    "transit": "PRESTO TRANSIT",
    "travel": "BOOKING.COM TRAVEL",
    "uncategorized": "LOCAL MERCHANT",
    "utilities": "PAYMENTUS UTILITIES",
    "vape": "RELX QLAB",
    "vending": "COCA COLA VENDING",
}


@dataclass
class AnonymizationReport:
    """Summary of one anonymization run."""

    input_path: str
    output_path: str
    rows_written: int
    validation_errors: int
    unique_descriptions: int
    unique_locations: int
    date_shift_days: int


def _normalize_text(value: str | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def _stable_int(value: str, modulo: int) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def _amount_multiplier(description_key: str) -> float:
    """Deterministic per-merchant multiplier so recurring charges stay recurring."""
    step = _stable_int(description_key, 17)  # 0..16
    return 0.88 + (step * 0.015)  # 0.88 .. 1.12


def _shift_dates(df: pd.DataFrame, target_start: str = "2025-01-03") -> pd.Series:
    """Shift all dates by one fixed delta so relative spacing is preserved."""
    min_date = df["date"].min()
    delta = pd.Timestamp(target_start) - min_date
    return df["date"] + delta


def _province_suffix(location: str) -> str:
    parts = location.upper().split()
    if parts and len(parts[-1]) == 2 and parts[-1].isalpha():
        return parts[-1]
    return "ON"


def _anonymize_locations(values: pd.Series) -> dict[str, str | None]:
    """Map each unique sub-description to a deterministic generic location."""
    mapping: dict[str, str | None] = {"": None}

    labels = [
        "Harbor",
        "Maple",
        "River",
        "North",
        "Cedar",
        "King",
        "Summit",
        "Union",
        "Garden",
        "Metro",
        "Crown",
        "Forest",
    ]

    for raw_value in sorted({_normalize_text(v) for v in values.dropna().tolist()}):
        if not raw_value:
            continue
        if "online" in raw_value:
            mapping[raw_value] = "Online"
            continue
        if "canada" in raw_value and len(raw_value.split()) <= 2:
            mapping[raw_value] = "Canada"
            continue

        province = _province_suffix(raw_value)
        label = labels[_stable_int(raw_value, len(labels))]
        code = _stable_int(raw_value, 90) + 10
        mapping[raw_value] = f"{label} City {code:02d} {province}"

    return mapping


def _build_description_mapping(df: pd.DataFrame) -> dict[str, str]:
    """Assign each unique merchant string a safe category-preserving alias."""
    counters: defaultdict[str, int] = defaultdict(int)
    mapping: dict[str, str] = {}

    unique_pairs = (
        df[["description", "category"]]
        .drop_duplicates()
        .sort_values(["category", "description"], kind="stable")
        .itertuples(index=False)
    )

    for description, category in unique_pairs:
        key = _normalize_text(description)
        if key in mapping:
            continue

        base = CATEGORY_BASE_MERCHANT.get(category, CATEGORY_BASE_MERCHANT["uncategorized"])
        counters[category] += 1
        mapping[key] = f"{base} {counters[category]:02d}"

    return mapping


def anonymize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return an anonymized DataFrame while preserving Scotia parser compatibility."""
    work = df.copy()
    work["anon_date"] = _shift_dates(work)

    location_map = _anonymize_locations(work["sub_description"])
    description_map = _build_description_mapping(work)

    work["anon_description"] = work["description"].map(
        lambda value: description_map[_normalize_text(value)]
    )
    work["anon_sub_description"] = work["sub_description"].map(
        lambda value: location_map[_normalize_text(value)]
    )

    def transform_amount(row: pd.Series) -> float:
        multiplier = _amount_multiplier(_normalize_text(row["description"]))
        shifted = round(float(row["amount"]) * multiplier, 2)
        if row["txn_type"] == "credit":
            return -abs(shifted)
        return abs(shifted)

    work["anon_amount"] = work.apply(transform_amount, axis=1)
    return work


def to_scotia_csv_shape(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the anonymized working frame back into a Scotia-shaped CSV frame."""
    out = pd.DataFrame(
        {
            "Filter": ["All available transactions"] + [""] * (len(df) - 1),
            "Date": df["anon_date"].dt.strftime("%Y-%m-%d"),
            "Description": df["anon_description"],
            "Sub-description": df["anon_sub_description"].fillna(""),
            "Status": df["status"].str.title(),
            "Type of Transaction": df["txn_type"].str.title(),
            "Amount": df["anon_amount"].map(lambda value: f"{value:.2f}"),
        }
    )
    return out[SCOTIA_COLUMNS]


def anonymize_sample_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_start: str = "2025-01-03",
) -> AnonymizationReport:
    """Read a real Scotia CSV and write an anonymized Scotia-shaped sample CSV."""
    raw_df, errors = load_transactions(input_path)
    agent_df = prepare_dataframe(raw_df)
    anonymized = anonymize_dataframe(agent_df)

    if not anonymized.empty:
        # Re-run the shift explicitly so CLI callers can override the anchor date.
        anonymized["anon_date"] = _shift_dates(anonymized, target_start=target_start)

    output_df = to_scotia_csv_shape(anonymized)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return AnonymizationReport(
        input_path=str(input_path),
        output_path=str(output_path),
        rows_written=len(output_df),
        validation_errors=len(errors),
        unique_descriptions=anonymized["description"].nunique(),
        unique_locations=anonymized["sub_description"].fillna("").nunique(),
        date_shift_days=(
            int((anonymized["anon_date"].min() - anonymized["date"].min()).days)
            if not anonymized.empty
            else 0
        ),
    )


def main() -> None:
    """CLI entrypoint for generating a public anonymized sample CSV."""
    parser = argparse.ArgumentParser(description="Create an anonymized Scotia sample CSV.")
    parser.add_argument("--input", required=True, help="Path to the real Scotia CSV export.")
    parser.add_argument(
        "--output",
        default="data/sample_anonymized.csv",
        help="Where to write the anonymized sample CSV.",
    )
    parser.add_argument(
        "--target-start",
        default="2025-01-03",
        help="Shift the earliest transaction date to this YYYY-MM-DD anchor.",
    )
    args = parser.parse_args()

    report = anonymize_sample_csv(args.input, args.output, target_start=args.target_start)
    print(f"Wrote anonymized sample to {report.output_path}")
    print(f"Rows written: {report.rows_written}")
    print(f"Validation errors excluded: {report.validation_errors}")
    print(f"Unique merchants anonymized: {report.unique_descriptions}")
    print(f"Unique locations anonymized: {report.unique_locations}")
    print(f"Date shift: {report.date_shift_days} day(s)")


if __name__ == "__main__":
    main()
