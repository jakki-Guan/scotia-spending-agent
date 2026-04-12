"""Shared pytest fixtures and markers."""

from pathlib import Path

import pytest

from scotia_agent.parser import load_transactions

REAL_CSV = Path(__file__).parent.parent / "data/raw/Scene_Visa_card_9024_041126.csv"


def pytest_configure(config):
    """Register custom markers to silence PytestUnknownMarkWarning."""
    config.addinivalue_line(
        "markers",
        "real_data: test requires the real Scotia csv (auto-skipped if missing).",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip real_data tests when the csv isn't present."""
    if REAL_CSV.exists():
        return
    skip_real = pytest.mark.skip(reason=f"Real data not available at {REAL_CSV}")
    for item in items:
        if "real_data" in item.keywords:
            item.add_marker(skip_real)


@pytest.fixture(scope="session")
def real_data():
    """Load the real Scotia csv once per session."""
    if not REAL_CSV.exists():
        pytest.skip(f"Real data not available at {REAL_CSV}")
    return load_transactions(REAL_CSV)
