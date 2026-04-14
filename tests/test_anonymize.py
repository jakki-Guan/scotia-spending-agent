from __future__ import annotations

from pathlib import Path

from scotia_agent.anonymize import anonymize_sample_csv
from scotia_agent.parser import load_transactions


def write_scotia_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    lines = ["Filter,Date,Description,Sub-description,Status,Type of Transaction,Amount"]
    for i, row in enumerate(rows):
        filter_value = "All available transactions" if i == 0 else ""
        lines.append(
            f'"{filter_value}","{row["date"]}","{row["description"]}",'
            f'"{row["sub_description"]}","{row["status"]}","{row["txn_type"]}",'
            f'"{row["amount"]}"'
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path


def test_anonymize_sample_csv_round_trips_through_parser(tmp_path):
    input_path = write_scotia_csv(
        tmp_path / "realish.csv",
        [
            {
                "date": "2026-01-10",
                "description": "tim hortons #2308",
                "sub_description": "Toronto ON",
                "status": "Posted",
                "txn_type": "Debit",
                "amount": "6.09",
            },
            {
                "date": "2026-01-15",
                "description": "spotify usa",
                "sub_description": "Online",
                "status": "Posted",
                "txn_type": "Debit",
                "amount": "11.99",
            },
            {
                "date": "2026-01-20",
                "description": "payment from chequing",
                "sub_description": "",
                "status": "Posted",
                "txn_type": "Credit",
                "amount": "-200.00",
            },
        ],
    )

    output_path = tmp_path / "sample_anonymized.csv"
    report = anonymize_sample_csv(input_path, output_path)

    assert output_path.exists()
    assert report.rows_written == 3

    parsed_df, errors = load_transactions(output_path)
    assert errors == []
    assert len(parsed_df) == 3

    descriptions = parsed_df["description"].str.lower().tolist()
    assert "tim hortons #2308" not in descriptions
    assert any("tim hortons cafe" in desc for desc in descriptions)
    assert any("spotify subscription" in desc for desc in descriptions)
    assert any("payment from chequing" in desc for desc in descriptions)

    sub_descriptions = parsed_df["sub_description"].fillna("").tolist()
    assert "Toronto ON" not in sub_descriptions
    assert "Online" in sub_descriptions

    assert parsed_df["txn_type"].tolist() == ["debit", "debit", "credit"]
    assert parsed_df["status"].tolist() == ["posted", "posted", "posted"]


def test_anonymize_sample_csv_preserves_relative_date_spacing(tmp_path):
    input_path = write_scotia_csv(
        tmp_path / "dates.csv",
        [
            {
                "date": "2026-03-01",
                "description": "tim hortons",
                "sub_description": "Hamilton ON",
                "status": "Posted",
                "txn_type": "Debit",
                "amount": "5.00",
            },
            {
                "date": "2026-03-11",
                "description": "tim hortons",
                "sub_description": "Hamilton ON",
                "status": "Posted",
                "txn_type": "Debit",
                "amount": "7.00",
            },
        ],
    )

    output_path = tmp_path / "sample_anonymized.csv"
    anonymize_sample_csv(input_path, output_path, target_start="2025-01-03")

    parsed_df, errors = load_transactions(output_path)
    assert errors == []
    assert parsed_df["date"].min().strftime("%Y-%m-%d") == "2025-01-03"
    assert (parsed_df["date"].iloc[1] - parsed_df["date"].iloc[0]).days == 10
