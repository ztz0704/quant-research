"""Independent coverage/availability audit. A vendor flag/hash is not PIT proof."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_alpha.data.research_contract import ASOF_FIELDS, digest_file, quality_report

EVIDENCE_KINDS = {
    "official_calendar",
    "security_lifecycle_registry",
    "historical_membership",
    "publication_history",
    "revision_history",
    "corporate_action_cash_and_shares",
}


def load_primary_evidence(root: Path, metadata: dict, date_end: str) -> dict:
    """Read only explicitly SHA-bound audit records before the declared boundary."""
    result = {}
    for kind, info in metadata.get("audit_files", {}).items():
        name = info["filename"]
        if Path(name).name != name or digest_file(root / name) != info["sha256"]:
            raise ValueError("Stale/unsafe primary audit evidence file")
        filters = [("date", "<=", pd.Timestamp(date_end))] if kind == "publication_log" else None
        result[kind] = pd.read_parquet(root / name, filters=filters)
    return result


def audit_pit_v3(
    panel: pd.DataFrame,
    metadata: dict,
    publication_log: pd.DataFrame | None = None,
    lifecycle_registry: pd.DataFrame | None = None,
) -> dict:
    """Require independently reviewed corroborating records beyond status strings.

    This verifies internal coverage and log reconciliation, not the authenticity
    of vendor testimony. Human review of primary publication/revision evidence
    remains mandatory; missing corroboration always blocks formal acceptance.
    """
    base = quality_report(panel, metadata)
    member = panel.universe_member.eq(True).fillna(False)
    quoted = panel.quote_observed
    native_trade = quoted & panel.buyable.eq(True).fillna(False)
    annual, transitions = [], []
    for year, group in panel.groupby(panel.date.dt.year):
        daily = (
            group.loc[native_trade.loc[group.index]]
            .groupby("date")
            .asset.nunique()
            .reindex(sorted(group.date.unique()), fill_value=0)
        )
        annual.append(
            {
                "year": year,
                "observed_assets": group.loc[quoted.loc[group.index]].asset.nunique(),
                "member_assets": group.loc[member.loc[group.index]].asset.nunique(),
                "tradeable_assets_min": daily.min() if len(daily) else 0,
                "tradeable_assets_median": daily.median() if len(daily) else 0,
                "quote_missing_rows": int(
                    (member.loc[group.index] & ~quoted.loc[group.index]).sum()
                ),
                "unknown_suspension_rows": int(group.suspended.isna().sum()),
                "calendar_sessions": group.date.nunique(),
                "sessions_without_native_buyable_quotes": int(daily.eq(0).sum()),
                "known_halt_rows": int(group.suspended.eq(True).fillna(False).sum()),
                "ipo_assets": group.loc[group.list_date.dt.year.eq(year), "asset"].nunique(),
                "delisted_assets": group.loc[group.delist_date.dt.year.eq(year), "asset"].nunique(),
            }
        )
    for asset, part in panel.sort_values(["asset", "date"]).groupby("asset"):
        industry = part.industry.astype("string")
        prior = part.universe_member.shift(1)
        changed = part.universe_member.notna() & prior.notna() & part.universe_member.ne(prior)
        factor_change = part.adj_factor.div(part.adj_factor.shift(1)).sub(1).abs().gt(1e-10)
        halts = part.suspended.eq(True).fillna(False)
        resumes = part.suspended.eq(False).fillna(False) & halts.shift(1, fill_value=False)
        transitions.append(
            {
                "asset": asset,
                "membership_changes": int(changed.sum()),
                "unknown_membership_rows": int(part.universe_member.isna().sum()),
                "halt_rows": int(halts.sum()),
                "resumption_events": int(resumes.sum()),
                "adjustment_changes": int(factor_change.sum()),
                "industry_changes": int(
                    (
                        industry.ne(industry.shift(1))
                        & industry.notna()
                        & industry.shift(1).notna()
                    ).sum()
                ),
            }
        )
    missing = pd.DataFrame(
        [
            {
                "field": c,
                "missing_rows": int(panel[c].isna().sum()),
                "missing_rate": float(panel[c].isna().mean()),
                "member_missing_rate": float(panel.loc[member, c].isna().mean()),
            }
            for c in panel.columns
        ]
    )
    blockers = list(base["blockers"])
    reviews = metadata.get("independent_evidence_reviews", [])
    reviewed = {
        r.get("kind")
        for r in reviews
        if r.get("reviewer")
        and r.get("primary_record_uri")
        and r.get("reviewed_at")
        and r.get("result") == "corroborated"
    }
    blockers.extend(
        f"missing_independent_evidence:{kind}" for kind in sorted(EVIDENCE_KINDS - reviewed)
    )
    log_issues = []
    if publication_log is None:
        blockers.append("no_primary_publication_log")
    else:
        required = {"date", "asset", "asof_field", "published_at", "revision_version"}
        if required - set(publication_log):
            raise ValueError("Publication evidence log schema mismatch")
        if publication_log.duplicated(["date", "asset", "asof_field"]).any():
            raise ValueError("Ambiguous publication log; materialize the actual as-of revision")
        if publication_log.revision_version.isna().any():
            blockers.append("missing_publication_revision_version")
        for asof in ASOF_FIELDS:
            evidence = publication_log.loc[publication_log.asof_field.eq(asof)]
            joined = panel.loc[member, ["date", "asset", asof]].merge(
                evidence, on=["date", "asset"], how="left", validate="one_to_one"
            )
            published = pd.to_datetime(joined.published_at, errors="raise")
            bad = published.isna() | joined[asof].isna() | ~published.eq(joined[asof])
            if bad.any():
                blockers.append(f"publication_log_mismatch:{asof}")
            log_issues.append(
                {"asof_field": asof, "rows": len(joined), "mismatch_rows": int(bad.sum())}
            )
    if lifecycle_registry is None:
        blockers.append("no_independent_lifecycle_registry")
    else:
        if {"asset", "list_date", "delist_date"} - set(lifecycle_registry):
            raise ValueError("Lifecycle registry schema mismatch")
        if lifecycle_registry.asset.duplicated().any():
            raise ValueError("Duplicate registry security identifiers")
        if lifecycle_registry.list_date.isna().any():
            blockers.append("missing_primary_registry_listing_dates")
        declared_delisted = set(
            lifecycle_registry.loc[lifecycle_registry.delist_date.notna(), "asset"]
        )
        expected = lifecycle_registry.loc[
            pd.to_datetime(lifecycle_registry.list_date).le(panel.date.max())
            & (
                lifecycle_registry.delist_date.isna()
                | pd.to_datetime(lifecycle_registry.delist_date).ge(panel.date.min())
            ),
            "asset",
        ]
        missing_assets = set(expected) - set(panel.asset)
        if missing_assets:
            blockers.append(f"lifecycle_coverage_missing_assets:{len(missing_assets)}")
        if declared_delisted and not declared_delisted.intersection(panel.asset):
            blockers.append("no_delisted_security_coverage")
    inconsistent = panel.buyable.eq(True).fillna(False) & (
        panel.suspended.eq(True).fillna(False) | panel.volume.le(0) | ~quoted
    )
    if inconsistent.any():
        blockers.append("inconsistent_buyability_quote_volume_or_halt")
    if not np.isfinite(panel.loc[quoted, "research_open"]).all():
        blockers.append("missing_research_adjustment_on_native_quote")
    action_changes = int(pd.DataFrame(transitions).adjustment_changes.sum())
    if action_changes:
        ordered = panel.sort_values(["asset", "date"])
        previous = ordered.groupby("asset").adj_factor.shift(1)
        changes = ordered.adj_factor.div(previous).sub(1).abs().gt(1e-10)
        actions = ordered.loc[changes]
        fields = {"action_complete", "action_id", "split_ratio", "cash_dividend", "action_asof"}
        if fields - set(actions):
            blockers.append("adjustment_changes_without_cash_share_event_records")
        else:
            complete_actions = (
                actions.action_complete.eq(True).fillna(False)
                & actions.action_id.notna()
                & actions.split_ratio.gt(0)
                & actions.cash_dividend.ge(0)
                & actions.action_asof.notna()
                & pd.to_datetime(actions.action_asof).le(actions.decision_time)
            )
            if not complete_actions.all():
                blockers.append("incomplete_adjustment_cash_share_event_coverage")
    report = {
        **base,
        "audit_version": "pit-audit-v3",
        "formal_ready": not blockers,
        "blockers": sorted(set(blockers)),
        "publication_check": log_issues,
        "trade_state_inconsistencies": int(inconsistent.sum()),
        "adjustment_change_events": action_changes,
        "evidence_grade": "internally_checked_requires_external_corroboration",
        "automatic_vendor_pit_certification": False,
    }
    return {
        "report": report,
        "annual_coverage": pd.DataFrame(annual),
        "field_missingness": missing,
        "lifecycle_state_changes": pd.DataFrame(transitions),
    }
