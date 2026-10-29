"""Build causal features and future mid-price labels with a coverage audit."""

import json

from market_microstructure.data.loader import load_orderbook
from market_microstructure.features.feature_pipeline import build_features, feature_columns
from market_microstructure.labels.labeling import build_labels
from market_microstructure.utils.config import script_config
from market_microstructure.utils.logging import configure_logging
from market_microstructure.utils.paths import research_path


def main() -> None:
    """Persist all rows; record warmup and unavailable future-label coverage."""
    config = script_config(__doc__)
    logger = configure_logging(config["logging"]["level"])
    book, _ = load_orderbook(research_path(config, config["data"]["input_path"]))
    features = build_features(book, **config["features"])
    labels = build_labels(features, **config["labels"])
    result = features.join(labels)
    result.to_parquet(research_path(config, config["data"]["features_path"]), index=False)
    columns = feature_columns(features)
    audit = {"input_rows": len(book), "output_rows": len(result), "feature_columns": columns,
             "feature_nan_rows": int(result[columns].isna().any(axis=1).sum()),
             "unavailable_label_rows": int(labels.future_return.isna().sum())}
    research_path(config, "outputs/tables/feature_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    logger.info("Features built: %s", audit)


if __name__ == "__main__":
    main()
