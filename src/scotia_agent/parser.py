"""
Parser for Scotiabank Scene Visa CSV exports.

Responsibilities (and ONLY these — no categorization, no analytics):
  1. Read the raw CSV (handles BOM, the leaked 'Filter' metadata column,
     trailing whitespace, string-typed amounts).
  2. Validate every row against a Pydantic schema.
  3. Return a clean pandas DataFrame + a list of row-level errors.

Design notes:
  - Bad rows are NOT silently dropped. They are returned in `errors` so the
    caller (or a test) can decide what to do. Failing loudly beats silent data loss.
  - Sign convention: `amount` is kept as Scotia exports it.
      Debit  rows -> positive (money out, e.g. coffee = +5.42)
      Credit rows -> negative (money in,  e.g. refund  = -14.29)
    To compute "total spending", filter `txn_type == 'debit'` and sum `amount`.
    Do NOT just sum the column — the credits will cancel out part of the spend.
  - Categorization is a SEPARATE pipeline step. This module knows nothing
    about categories. Compose them at the call site:
        df, errs = load_transactions(path)
        df['category'] = df['description'].apply(categorize)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

# ---------- Schema ----------


class Transaction(BaseModel):
    """One validated row from a Scotia Scene Visa export."""

    date: date
    description: str = Field(min_length=1)
    sub_description: str | None = None
    status: Literal["posted", "pending"]
    txn_type: Literal["debit", "credit"]
    amount: float

    @field_validator("description", "sub_description", mode="before")
    @classmethod
    def _strip(cls, v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return str(v).strip() or None

    @field_validator("status", "txn_type", mode="before")
    @classmethod
    def _lower(cls, v):
        return str(v).strip().lower()


# ---------- Loader ----------

# Scotia exports use these exact headers; we rename to snake_case internally.
COLUMN_MAP = {
    "Date": "date",
    "Description": "description",
    "Sub-description": "sub_description",
    "Status": "status",
    "Type of Transaction": "txn_type",
    "Amount": "amount",
}


def load_transactions(
    path: str | Path,
) -> tuple[pd.DataFrame, list[dict]]:
    """Load and validate a Scotia Scene Visa CSV.

    Returns:
        (clean_df, errors)
        - clean_df: DataFrame of validated rows, sorted by date ascending.
                    Columns: date, description, sub_description, status,
                             txn_type, amount.
        - errors:   list of {"row_index": int, "errors": [...], "raw": {...}}
                    for any rows that failed validation.

    Raises:
        FileNotFoundError: if `path` does not exist.
        ValueError: if expected columns are missing from the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    # encoding='utf-8-sig' strips the BOM that Scotia puts on the first header.
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str)

    # The 'Filter' column is metadata about the export (date range), not data.
    # It only has a value on row 0. Drop it if present.
    raw = raw.drop(columns=[c for c in raw.columns if c.strip() == "Filter"], errors="ignore")

    missing = [k for k in COLUMN_MAP if k not in raw.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}. Got: {list(raw.columns)}")

    raw = raw.rename(columns=COLUMN_MAP)

    valid_rows: list[dict] = []
    errors: list[dict] = []

    for idx, row in raw.iterrows():
        try:
            txn = Transaction(
                date=row["date"],
                description=row["description"],
                sub_description=row.get("sub_description"),
                status=row["status"],
                txn_type=row["txn_type"],
                amount=row["amount"],
            )
            valid_rows.append(txn.model_dump())
        except ValidationError as e:
            errors.append(
                {
                    "row_index": int(idx),
                    "errors": e.errors(),
                    "raw": row.to_dict(),
                }
            )

    df = pd.DataFrame(valid_rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    return df, errors


# ---------- Convenience helpers (cheap; keep them here so callers don't reinvent) ----------


def spending_only(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that represent actual outgoing money (debits, posted)."""
    return df[(df["txn_type"] == "debit") & (df["status"] == "posted")].copy()


def total_spend(df: pd.DataFrame) -> float:
    """Sum of debit amounts. Use this instead of df['amount'].sum()."""
    return float(spending_only(df)["amount"].sum())
