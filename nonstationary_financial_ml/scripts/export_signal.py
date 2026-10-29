"""Export historical OOS scores with timing metadata; never call them current recommendations."""
import argparse

from financial_ml.evaluation import load_research
from financial_ml.utils import config, output, sha256, write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model", default="ridge")
    parser.add_argument("--prediction", choices=["raw_prediction", "neutralized_prediction"], default="raw_prediction")
    args = parser.parse_args()
    cfg = config(args.config)
    _, predictions, _ = load_research(cfg)
    chosen = predictions.loc[predictions.model.eq(args.model)]
    if chosen.empty:
        raise ValueError("No predictions for declared model")
    signal = chosen[["date", "asset", args.prediction]].rename(columns={args.prediction: "alpha"})
    target = output(cfg) / f"predictions/{args.model}_{args.prediction}_signal.csv"
    signal.to_csv(target, index=False)
    write_json({"model": args.model, "prediction": args.prediction, "horizon": cfg["labels"]["forward_horizon"],
                "first_date": str(signal.date.min()), "last_date": str(signal.date.max()), "rows": len(signal),
                "semantics": "uncalibrated regression return forecast, treated as a score by the portfolio bridge",
                "decision_timing": "features through t-1; close(t) to close(t+h) label",
                "historical_oos_only": True, "full_risk_panel_coverage": False, "sha256": sha256(target)}, target.with_suffix(".json"))
    print(target)
