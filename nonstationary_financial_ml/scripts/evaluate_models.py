"""Write all-model IC, errors, quantile, turnover and modeled-cost diagnostics."""
import argparse

from financial_ml.evaluation import evaluate
from financial_ml.utils import config, logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = config(args.config)
    frames = evaluate(cfg)
    logger(cfg, "evaluate_models").info("Evaluated %s model/prediction variants", len(frames["model_performance"]))
