"""Create a reproducible synthetic panel or validate an existing supplier panel."""

import argparse
import json

from risk_aware_portfolio.data.loader import load_panel
from risk_aware_portfolio.data.synthetic import generate_synthetic
from risk_aware_portfolio.data.validation import validate_panel
from risk_aware_portfolio.utils.config import load_config
from risk_aware_portfolio.utils.logging import configure_logging
from risk_aware_portfolio.utils.manifest import file_hash
from risk_aware_portfolio.utils.paths import ensure_directories, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--existing", action="store_true", help="Validate input; do not generate")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_directories()
    logger = configure_logging("prepare_data")
    path = resolve_path(config["data"]["input_path"])
    if args.existing:
        panel = load_panel(path, max_abs_return=config["data"]["max_abs_return"])
    else:
        data = generate_synthetic(config)
        panel = data.panel
        validate_panel(panel)
        path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(path, index=False)
        data.truth.to_parquet(resolve_path("data/raw/synthetic_truth.parquet"), index=False)
        data.factor_returns.to_parquet(resolve_path("data/raw/true_factor_returns.parquet"))
        data.factor_covariance.to_parquet(resolve_path("data/raw/true_factor_covariance.parquet"))
        metadata = {
            "panel_sha256": file_hash(path),
            "synthetic": config["synthetic"],
            "industries": config["risk_model"]["industries"],
            "styles": config["risk_model"]["styles"],
            "truth_sha256": file_hash(resolve_path("data/raw/synthetic_truth.parquet")),
            "factor_returns_sha256": file_hash(
                resolve_path("data/raw/true_factor_returns.parquet")
            ),
            "factor_covariance_sha256": file_hash(
                resolve_path("data/raw/true_factor_covariance.parquet")
            ),
        }
        resolve_path("data/raw/synthetic_manifest.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    logger.info(
        "Validated %d rows, %d dates, %d assets",
        len(panel),
        panel["date"].nunique(),
        panel["asset"].nunique(),
    )
    print(f"Validated panel: {len(panel):,} rows, {panel['date'].nunique()} dates")


if __name__ == "__main__":
    main()
