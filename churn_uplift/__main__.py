"""Command line: python -m churn_uplift [--config config.toml] {train,score,monitor} ..."""
import argparse
import sys
from pathlib import Path

from churn_uplift.pipeline import Config, load_bundle, load_training_data, monitor, read_table, score, train


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m churn_uplift", description=__doc__)
    parser.add_argument("--config", default="config.toml", help="settings file (default: config.toml)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train", help="train the models and save them under artifacts/")
    score_cmd = commands.add_parser("score", help="write offer recommendations for a customer file (CSV or parquet)")
    score_cmd.add_argument("customers")
    score_cmd.add_argument("--output", default="outputs/scored_customers.csv")
    monitor_cmd = commands.add_parser("monitor", help="compare a customer file with the training data; exit 1 on drift")
    monitor_cmd.add_argument("customers")
    args = parser.parse_args(argv)
    config = Config.load(args.config)

    try:
        if args.command == "train":
            metrics = train(load_training_data(config), config)
            dr = metrics["validation_dr_qini"]
            print(f"Churn model: {metrics['churn_model']}   uplift model: {metrics['uplift_model']}")
            print(f"Validation DR Qini {dr['estimate']:+.2f} (95% CI {dr['low']:+.2f} to {dr['high']:+.2f}); "
                  f"offers {'enabled' if metrics['validated'] else 'disabled by the safety gate'}")
            print(f"Saved to {config.artifacts}/")
            return 0

        bundle = load_bundle(config)
        customers = read_table(args.customers)
        if args.command == "score":
            recs = score(bundle, customers, config)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            recs.to_csv(args.output, index=False)
            offers = int((recs["recommended_intervention"] == "Retention_Offer").sum())
            print(f"Scored {len(recs):,} customers, {offers:,} offers recommended -> {args.output}")
            return 0

        report, drifted = monitor(bundle, customers, config)
        Path(config.drift_report).parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(config.drift_report, index=False)
        flagged = report[report["status"] == "drift"]
        print(f"{len(flagged)} of {len(report)} columns drifted (PSI >= {config.psi_threshold}) -> {config.drift_report}")
        for _, row in flagged.head(10).iterrows():
            print(f"  {row['column']:<20} PSI {row['psi']:.3f}")
        return 1 if drifted else 0
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
