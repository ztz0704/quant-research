"""Train all declared models with temporal validation-only tuning."""
import argparse
import json

import pandas as pd

from financial_ml.training import train_all
from financial_ml.utils import config, output, path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = config(args.config)
    manifest = json.loads((output(cfg) / "tables/feature_manifest.json").read_text(encoding="utf-8"))
    train_all(pd.read_parquet(path(cfg["data"]["features_path"])), manifest["feature_names"], cfg)
