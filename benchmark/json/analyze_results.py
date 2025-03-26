import os
import argparse
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from benchmark.util import analyze_directory  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze json benchmark results across multiple methods."
    )
    parser.add_argument(
        "--dirs",
        type=str,
        nargs="+",
        required=True,
        help="Directories containing result files to analyze.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap samples for confidence intervals.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for intervals (between 0 and 1).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="analysis_results.csv",
        help="File to write analysis results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    results_list = []
    for directory in args.dirs:
        for sub_dir in os.listdir(directory):
            print(f"Analyzing {sub_dir}...")
            full_path = os.path.join(directory, sub_dir)
            result = analyze_directory(
                full_path, n_bootstrap=args.n_bootstrap, confidence=args.confidence
            )
            if result:
                results_list.append(result)

    df = pd.DataFrame(results_list)
    df.to_csv(args.output_file, index=False)

    print(f"Results written to {args.output_file}")

    print(f"Analysis Results (Bootstrap CI: {args.confidence * 100}%)\n")
    print("=" * 80 + "\n\n")

    for result in results_list:
        print(f"Directory: {result['directory']}\n")
        print(f"Number of samples: {result['n_samples']}\n")
        print(
            f"Accuracy: {result['mean_accuracy']:.3f} ({result['accuracy_ci'][0]:.2f}, {result['accuracy_ci'][1]:.2f})\n"
        )
        print(
            f"Inference time: {result['mean_inference_time']:.2f} ({result['mean_inference_time_low']:.2f}, {result['mean_inference_time_high']:.2f})\n"
        )
        print("\n" + "-" * 80 + "\n\n")


if __name__ == "__main__":
    main()
