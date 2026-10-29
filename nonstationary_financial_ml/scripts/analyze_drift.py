"""Analyze feature/concept drift, model age and actual refitting experiments."""
import argparse

from financial_ml.cadence import study
from financial_ml.drift import analyze
from financial_ml.utils import config, logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = config(args.config)
    analyze(cfg)
    study(cfg)
    logger(cfg, "analyze_drift").info("Feature/concept/age diagnostics and complete cadence grid saved")
