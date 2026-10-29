"""Validate and snapshot the declared real source; synthetic fallback is forbidden."""
import argparse

from financial_ml.data import prepare
from financial_ml.utils import config, logger

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/base.yaml")
if __name__ == "__main__":
    args = parser.parse_args()
    cfg = config(args.config)
    frame = prepare(cfg)
    logger(cfg, "prepare_data").info("Validated %s rows, %s assets; source=%s", len(frame), frame.asset.nunique(), cfg["data"]["source"])
