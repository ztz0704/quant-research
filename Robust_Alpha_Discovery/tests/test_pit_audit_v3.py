import copy

import pandas as pd
import pytest

from robust_alpha.data.pit_audit_v3 import audit_pit_v3
from robust_alpha.data.research_contract import local_session_time, normalize_dynamic_panel
from robust_alpha.data.research_fixture import engineering_sample


def test_missing_object_industry_is_reported_without_ambiguous_na():
    raw, calendar, metadata = engineering_sample(days=130, assets=10)
    raw["industry"] = pd.NA
    result = audit_pit_v3(normalize_dynamic_panel(raw, calendar), metadata)
    assert result["lifecycle_state_changes"].industry_changes.eq(0).all()
    assert not result["report"]["formal_ready"]
    missing = result["field_missingness"].set_index("field")
    assert missing.loc["industry", "missing_rate"] == 1


def test_vendor_verified_flags_do_not_create_pit_certification():
    raw, calendar, metadata = engineering_sample(days=130, assets=10)
    metadata = copy.deepcopy(metadata)
    for field in metadata["fields"].values():
        field.update(status="verified_pit", source="claimed", version="claimed")
    metadata.update(
        official_calendar=True, survivorship_free=True, publication_history_verified=True
    )
    result = audit_pit_v3(normalize_dynamic_panel(raw, calendar), metadata)
    assert not result["report"]["formal_ready"]
    assert "no_primary_publication_log" in result["report"]["blockers"]
    assert not result["report"]["automatic_vendor_pit_certification"]
    assert len(result["annual_coverage"]) == 1
    assert len(result["lifecycle_state_changes"]) == 10


def test_registry_missing_delisted_names_and_mismatched_publications_block():
    raw, calendar, metadata = engineering_sample(days=130, assets=10)
    panel = normalize_dynamic_panel(raw, calendar)
    registry = pd.DataFrame(
        {
            "asset": ["MISSING_DELISTED"],
            "list_date": [panel.date.min()],
            "delist_date": [panel.date.max()],
        }
    )
    log = pd.DataFrame(columns=["date", "asset", "asof_field", "published_at", "revision_version"])
    result = audit_pit_v3(panel, metadata, log, registry)
    assert "lifecycle_coverage_missing_assets:1" in result["report"]["blockers"]
    assert "no_delisted_security_coverage" in result["report"]["blockers"]
    assert any(s.startswith("publication_log_mismatch") for s in result["report"]["blockers"])
    with pytest.raises(ValueError, match="schema"):
        audit_pit_v3(panel, metadata, pd.DataFrame({"date": []}), registry)


def test_calendar_zero_quote_day_is_counted_and_wall_time_preserves_nat():
    raw, calendar, metadata = engineering_sample(days=130, assets=10, continuous=True)
    day = raw.date.unique()[70]
    mask = raw.date.eq(day)
    raw.loc[mask, ["raw_open", "raw_close", "raw_high", "raw_low", "volume", "amount"]] = float(
        "nan"
    )
    raw.loc[mask, ["buyable", "sellable"]] = False
    result = audit_pit_v3(normalize_dynamic_panel(raw, calendar), metadata)
    annual = result["annual_coverage"].iloc[0]
    assert annual.tradeable_assets_min == 0
    assert annual.sessions_without_native_buyable_quotes == 1
    assert annual.calendar_sessions == 130
    times = local_session_time([pd.Timestamp("2020-01-02"), pd.NaT], 9, 30)
    assert times[0] == pd.Timestamp("2020-01-02 09:30")
    assert pd.isna(times[1])
