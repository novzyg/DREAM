import argparse
import json
from pathlib import Path


DEFAULT_METRICS = (
    "ddi_rate",
    "jaccard",
    "prauc",
    "avg_prc",
    "avg_recall",
    "avg_f1",
    "avg_med",
)


def latest_summary_file(reports_dir):
    files = sorted(reports_dir.glob("cross_seed_summary_*.json"))
    return files[-1] if files else None


def extract_summary(path, metrics, include_n=False, include_source=False):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"{path} does not contain a summary object")

    extracted = {}
    for metric in metrics:
        item = summary.get(metric)
        if not isinstance(item, dict):
            continue

        metric_data = {}
        if "mean" in item:
            metric_data["mean"] = item["mean"]
        if "std" in item:
            metric_data["std"] = item["std"]
        if include_n and "n" in item:
            metric_data["n"] = item["n"]

        extracted[metric] = metric_data

    if include_source:
        extracted["_source"] = str(path)

    return extracted


def collect_results(results_dir, metrics, include_n=False, include_source=False):
    output = {}
    warnings = []

    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        model_results = {}

        for dataset_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            reports_dir = dataset_dir / "reports"
            if not reports_dir.is_dir():
                continue

            summary_path = latest_summary_file(reports_dir)
            if summary_path is None:
                continue

            try:
                model_results[dataset_dir.name] = extract_summary(
                    summary_path,
                    metrics,
                    include_n=include_n,
                    include_source=include_source,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                warnings.append(f"{summary_path}: {exc}")

        if model_results:
            output[model_dir.name] = model_results

    return output, warnings


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    default_results_dir = project_root / "results"
    default_output = default_results_dir / "structured_results_summary.json"

    parser = argparse.ArgumentParser(
        description="Export cross-seed experiment summaries as a structured JSON file."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir,
        help=f"Results directory to scan. Default: {default_results_dir}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output JSON path. Default: {default_output}",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metrics to export from each summary object.",
    )
    parser.add_argument(
        "--include-n",
        action="store_true",
        help="Include the n field for each metric when present.",
    )
    parser.add_argument(
        "--include-source",
        action="store_true",
        help="Include the selected cross_seed_summary JSON path for each dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_path = args.output.resolve()

    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")

    data, warnings = collect_results(
        results_dir,
        args.metrics,
        include_n=args.include_n,
        include_source=args.include_source,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Exported {len(data)} model(s) to {output_path}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
