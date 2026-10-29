"""Build date-local features and explicitly timed forward return labels."""
import argparse

from financial_ml.data import load
from financial_ml.features import build_features
from financial_ml.labels import add_labels
from financial_ml.utils import config, logger, output, path, sha256, write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = config(args.config)
    panel, names = build_features(load(path(cfg["data"]["input_path"])), cfg)
    panel = add_labels(panel, cfg["labels"]["forward_horizon"])
    target = path(cfg["data"]["features_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(target, index=False)
    write_json({"feature_names": names, "rows": len(panel), "eligible_rows": int(panel.eligible.sum()),
                "valid_labels": int(panel.target_valid.sum()), "horizon": cfg["labels"]["forward_horizon"],
                "input_sha256": sha256(path(cfg["data"]["input_path"])), "features_sha256": sha256(target),
                "timing": "features through t-1; theoretical close(t) to close(t+h) return"}, output(cfg) / "tables/feature_manifest.json")
    logger(cfg, "build_features").info("Built %s causal features; eligible=%s", len(names), panel.eligible.sum())
