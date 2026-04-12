"""Tests for scotia_agent.parser.

Structure:
  - Synthetic CSV tests (hermetic, always run): schema, cleaning,
    error handling, sign convention, helpers.
  - Real data tests (marked real_data, skip if csv missing): verify
    the parser holds up against Jake's actual 1000-row export.
"""

from __future__ import annotations

import pytest

from scotia_agent.parser import (
    Transaction,
    load_transactions,
    spending_only,
    total_spend,
)

# ---------------------------------------------------------------------------
# Helper: build a Scotia-shaped csv from a list of row dicts.
# Writes the file with a BOM (like Scotia actually does).
# ---------------------------------------------------------------------------


def write_scotia_csv(tmp_path, rows):
    """`rows` is a list of dicts with keys:
       date, desc, sub, status, txn_type, amount
    The Filter column gets one value on row 0, empty afterwards —
    this mimics Scotia's actual export format."""
    lines = ["Filter,Date,Description,Sub-description,Status,Type of Transaction,Amount"]
    for i, r in enumerate(rows):
        filter_val = "All available transactions" if i == 0 else ""
        lines.append(
            f'"{filter_val}","{r["date"]}","{r["desc"]}","{r["sub"]}",'
            f'"{r["status"]}","{r["txn_type"]}","{r["amount"]}"'
        )
    path = tmp_path / "test.csv"
    # utf-8-sig writes a BOM, same as Scotia's actual export.
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path


def row(
    date="2026-04-09",
    desc="coffee",
    sub="Hamilton On",
    status="posted",
    txn_type="Debit",
    amount="5.00",
):
    """Make a row dict with sensible defaults; override what you need."""
    return dict(date=date, desc=desc, sub=sub, status=status, txn_type=txn_type, amount=amount)


# ---------------------------------------------------------------------------
# Schema & cleaning (synthetic)
# ---------------------------------------------------------------------------


def test_loads_minimal_csv(tmp_path):
    path = write_scotia_csv(tmp_path, [row(desc="tim hortons #0226")])
    df, errors = load_transactions(path)
    assert len(df) == 1
    assert errors == []


def test_snake_case_columns(tmp_path):
    path = write_scotia_csv(tmp_path, [row()])
    df, _ = load_transactions(path)
    assert set(df.columns) == {
        "date",
        "description",
        "sub_description",
        "status",
        "txn_type",
        "amount",
    }


def test_date_is_datetime(tmp_path):
    path = write_scotia_csv(tmp_path, [row()])
    df, _ = load_transactions(path)
    assert df["date"].dtype.name.startswith("datetime")


def test_amount_is_float(tmp_path):
    path = write_scotia_csv(tmp_path, [row()])
    df, _ = load_transactions(path)
    assert df["amount"].dtype == "float64"


def test_strips_trailing_whitespace(tmp_path):
    path = write_scotia_csv(
        tmp_path,
        [
            row(desc="tim hortons    ", sub="Hamilton On   "),
        ],
    )
    df, _ = load_transactions(path)
    assert df["description"].iloc[0] == "tim hortons"
    assert df["sub_description"].iloc[0] == "Hamilton On"


def test_status_and_txn_type_lowercased(tmp_path):
    path = write_scotia_csv(tmp_path, [row(status="POSTED", txn_type="Debit")])
    df, _ = load_transactions(path)
    assert df["status"].iloc[0] == "posted"
    assert df["txn_type"].iloc[0] == "debit"


def test_bom_is_stripped(tmp_path):
    """Scotia exports with a BOM; utf-8-sig must strip it."""
    path = write_scotia_csv(tmp_path, [row()])
    df, _ = load_transactions(path)
    assert not any("\ufeff" in c for c in df.columns)


def test_filter_column_is_dropped(tmp_path):
    path = write_scotia_csv(tmp_path, [row()])
    df, _ = load_transactions(path)
    assert "filter" not in df.columns
    assert "Filter" not in df.columns


def test_sorted_by_date_ascending(tmp_path):
    path = write_scotia_csv(
        tmp_path,
        [
            row(date="2026-04-10", desc="b"),
            row(date="2026-04-08", desc="a"),
            row(date="2026-04-09", desc="c"),
        ],
    )
    df, _ = load_transactions(path)
    assert df["description"].tolist() == ["a", "c", "b"]


# ---------------------------------------------------------------------------
# Sign convention (synthetic)
# ---------------------------------------------------------------------------


def test_debit_amount_stays_positive(tmp_path):
    path = write_scotia_csv(tmp_path, [row(amount="5.00", txn_type="Debit")])
    df, _ = load_transactions(path)
    assert df["amount"].iloc[0] == 5.00


def test_credit_amount_stays_negative(tmp_path):
    """Scotia convention: refunds/payments come in as negative amounts."""
    path = write_scotia_csv(
        tmp_path,
        [
            row(desc="refund", txn_type="Credit", amount="-14.29"),
        ],
    )
    df, _ = load_transactions(path)
    assert df["amount"].iloc[0] == -14.29


# ---------------------------------------------------------------------------
# Error handling (synthetic)
# ---------------------------------------------------------------------------


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_transactions("nonexistent.csv")


def test_missing_required_column_raises(tmp_path):
    """If Scotia changes its format and drops a column, fail loudly."""
    path = tmp_path / "bad.csv"
    path.write_text("Date,Description,Amount\n2026-04-09,coffee,5.00\n")
    with pytest.raises(ValueError, match="missing expected columns"):
        load_transactions(path)


def test_bad_row_goes_to_errors_not_df(tmp_path):
    """Invalid txn_type should fail Pydantic validation and go to errors."""
    path = write_scotia_csv(
        tmp_path,
        [
            row(desc="good row", txn_type="Debit"),
            row(desc="bad row", txn_type="Teleport"),  # bad
        ],
    )
    df, errors = load_transactions(path)
    assert len(df) == 1
    assert len(errors) == 1
    assert df["description"].iloc[0] == "good row"


def test_empty_description_rejected(tmp_path):
    """description has min_length=1 in the schema."""
    path = write_scotia_csv(tmp_path, [row(desc="")])
    df, errors = load_transactions(path)
    assert len(df) == 0
    assert len(errors) == 1


# ---------------------------------------------------------------------------
# Helpers (synthetic)
# ---------------------------------------------------------------------------


def test_spending_only_excludes_credits(tmp_path):
    path = write_scotia_csv(
        tmp_path,
        [
            row(desc="coffee", txn_type="Debit", amount="5.00"),
            row(desc="refund", txn_type="Credit", amount="-10.00"),
        ],
    )
    df, _ = load_transactions(path)
    spend = spending_only(df)
    assert len(spend) == 1
    assert (spend["txn_type"] == "debit").all()


def test_total_spend_sums_only_debits(tmp_path):
    """Critical: credits are negative, so naive df.amount.sum() would
    understate spending. total_spend() must exclude credits."""
    path = write_scotia_csv(
        tmp_path,
        [
            row(desc="coffee", txn_type="Debit", amount="5.00"),
            row(desc="lunch", txn_type="Debit", amount="15.00"),
            row(desc="refund", txn_type="Credit", amount="-10.00"),
        ],
    )
    df, _ = load_transactions(path)
    assert total_spend(df) == 20.00  # NOT 10.00


# ---------------------------------------------------------------------------
# Pydantic schema direct tests (no file I/O)
# ---------------------------------------------------------------------------


def test_transaction_schema_normalizes_fields():
    txn = Transaction(
        date="2026-04-09",
        description="  Tim Hortons  ",  # should strip
        sub_description="Hamilton On",
        status="POSTED",  # should lowercase
        txn_type="Debit",
        amount="5.42",  # should coerce to float
    )
    assert txn.description == "Tim Hortons"
    assert txn.status == "posted"
    assert txn.txn_type == "debit"
    assert txn.amount == 5.42


def test_transaction_schema_rejects_unknown_status():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Transaction(
            date="2026-04-09",
            description="x",
            sub_description=None,
            status="weird",
            txn_type="debit",
            amount=5.00,
        )


# ---------------------------------------------------------------------------
# Real data (auto-skipped if csv missing — see conftest.py)
# ---------------------------------------------------------------------------


@pytest.mark.real_data
def test_real_data_loads_all_rows(real_data):
    df, errors = real_data
    assert len(df) == 1000
    assert len(errors) == 0


@pytest.mark.real_data
def test_real_data_no_whitespace(real_data):
    df, _ = real_data
    assert not df["description"].str.startswith(" ").any()
    assert not df["description"].str.endswith(" ").any()


@pytest.mark.real_data
def test_real_data_sign_convention(real_data):
    df, _ = real_data
    assert (df[df.txn_type == "debit"]["amount"] > 0).all()
    assert (df[df.txn_type == "credit"]["amount"] < 0).all()


@pytest.mark.real_data
def test_real_data_enum_values_constrained(real_data):
    df, _ = real_data
    assert set(df["txn_type"].unique()) <= {"debit", "credit"}
    assert set(df["status"].unique()) <= {"posted", "pending"}
