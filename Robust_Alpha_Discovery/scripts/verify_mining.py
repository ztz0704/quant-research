"""Verify real mining fingerprints, rebuild factors and mutate actual future observations."""

import argparse

import pandas as pd

from robust_alpha.config import project_path
from robust_alpha.data import add_eligibility, load_panel
from robust_alpha.features import compute_features
from robust_alpha.mining import prepare, read_selection, settings, write_json
from robust_alpha.preprocessing import cross_sectional_transform

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mine_real.yaml")
    args = parser.parse_args()
    cfg, destination = settings(args.config)
    frozen = read_selection(cfg, destination)
    panel, _ = prepare(cfg, frozen["history_through"], labels=False)
    columns = [
        "date",
        "asset",
        "eligible",
        *cfg["signals"],
        *(f"z__{name}" for name in cfg["signals"]),
    ]
    stored = pd.read_parquet(destination / "factor_values.parquet")
    pd.testing.assert_frame_equal(stored, panel[columns], check_exact=True)
    raw = load_panel(project_path(cfg["data"]["path"]), date_end=frozen["history_through"])
    dates = sorted(raw.date.unique())
    if len(dates) < 25:
        raise ValueError("Need 25 market sessions to verify a real future perturbation")
    cutoff = dates[-20]
    changed = raw.copy()
    future = changed.date.gt(cutoff)
    changed.loc[future, ["open", "high", "low", "close"]] *= 2.0
    changed.loc[future, ["volume", "amount"]] *= 3.0
    rebuilt = add_eligibility(changed, min_history=cfg["data"]["min_history"])
    rebuilt = compute_features(rebuilt, cfg["signals"], cfg.get("custom_signals"))
    rebuilt = cross_sectional_transform(
        rebuilt, cfg["signals"], method=cfg["preprocessing"]["method"]
    )
    pd.testing.assert_frame_equal(
        panel.loc[panel.date.le(cutoff), columns],
        rebuilt.loc[rebuilt.date.le(cutoff), columns],
        check_exact=True,
    )
    manifest = destination / "test_manifest.json"
    if manifest.is_file():
        import json

        from robust_alpha.mining import file_hash

        testing = json.loads(manifest.read_text(encoding="utf-8"))
        if testing["selection_sha256"] != file_hash(destination / "selection.json"):
            raise ValueError("Test results belong to a previous selection")
    write_json(
        destination / "mining_verification.json",
        {
            "status": "passed",
            "factor_count": len(cfg["signals"]),
            "rows_rebuilt": len(stored),
            "history_cutoff": str(pd.Timestamp(cutoff).date()),
            "future_perturbation": "actual cached prices/volume/amount changed; past factors exact",
            "selected": frozen["selected"],
            "code_and_artifact_hashes": "passed",
        },
    )
    print(f"Verified {len(cfg['signals'])} factors and {len(stored)} real rows; {destination}")
