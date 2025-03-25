import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_point,
    geom_text,
    labs,
    scale_y_log10,
    theme,
    element_text,
    geom_errorbarh,
    geom_errorbar,
    theme_light,
    scale_color_brewer,
    coord_cartesian,
    position_nudge,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze regex benchmark results across multiple methods."
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
    # parser.add_argument(
    #    "--plot",
    #    action="store_true",
    #    help="Generate plots of the results."
    # )
    return parser.parse_args()


def bootstrap_ci(data, statistic, n_bootstrap=1000, confidence=0.95, random_state=None):
    """Compute bootstrap confidence interval for a statistic."""
    rng = np.random.RandomState(random_state)
    bootstrap_samples = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrap_samples.append(statistic(sample))

    # Sort the bootstrap samples
    bootstrap_samples.sort()

    # Compute the confidence interval
    alpha = 1 - confidence
    lower_idx = int(n_bootstrap * (alpha / 2))
    upper_idx = int(n_bootstrap * (1 - alpha / 2))

    return bootstrap_samples[lower_idx], bootstrap_samples[upper_idx]


def analyze_directory(directory, n_bootstrap=1000, confidence=0.95):
    """Analyze all result files in a directory."""
    accuracies = []
    inference_times = []

    # Find all result files
    result_files = sorted(
        [f for f in os.listdir(directory) if f.endswith("_result.pkl")]
    )

    if not result_files:
        print(f"No result files found in {directory}")
        return None

    for file in result_files:
        file_path = os.path.join(directory, file)
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                metadata = data["metadata"]
                accuracies.append(metadata["accuracy"])
                inference_times.append(metadata["inference_time"])
        except (pickle.UnpicklingError, KeyError, FileNotFoundError) as e:
            print(f"Error processing {file_path}: {e}")

    if not accuracies:
        print(f"No valid data found in {directory}")
        return None

    # Calculate statistics
    mean_accuracy = np.mean(accuracies)
    mean_inference_time = np.mean(inference_times)

    # Bootstrap confidence intervals
    acc_ci = bootstrap_ci(
        accuracies, statistic=np.mean, n_bootstrap=n_bootstrap, confidence=confidence
    )

    time_ci = bootstrap_ci(
        inference_times,
        statistic=np.mean,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
    )

    stats_files = sorted([f for f in os.listdir(directory) if f.endswith("_stats.pkl")])
    stats = []
    for file in stats_files:
        file_path = os.path.join(directory, file)
        with open(file_path, "rb") as f:
            stats.append(pickle.load(f))

    with open(os.path.join(directory, "args.json"), "r") as f:
        args = json.load(f)
        args_info = {
            "lm": args["model_name"],
            "n_particles": args["n_particles"],
            "sampler_name": args["sampler_name"],
            "max_tokens": args["max_tokens"],
            "sampler_args": args["sampler_args"],
        }

    return {
        "directory": directory,
        "n_samples": len(accuracies),
        "mean_accuracy": mean_accuracy,
        "accuracy_ci": acc_ci,
        "mean_accuracy_low": acc_ci[0],
        "mean_accuracy_high": acc_ci[1],
        "mean_inference_time": mean_inference_time,
        "time_ci": time_ci,
        "mean_inference_time_low": time_ci[0],
        "mean_inference_time_high": time_ci[1],
        **args_info,
    }


def main():
    args = parse_args()

    results = {}
    for directory in args.dirs:
        print(f"Analyzing {directory}...")
        result = analyze_directory(
            directory, n_bootstrap=args.n_bootstrap, confidence=args.confidence
        )
        if result:
            results[directory] = result

    cols = [
        "directory",
        "n_samples",
        "mean_accuracy",
        "accuracy_ci",
        "mean_accuracy_low",
        "mean_accuracy_high",
        "mean_inference_time",
        "time_ci",
        "mean_inference_time_low",
        "mean_inference_time_high",
        "lm",
        "n_particles",
        "sampler_name",
        "max_tokens",
        "sampler_args",
    ]
    df = pd.DataFrame(results.values(), columns=cols)
    df.to_csv(args.output_file, index=False)

    print(f"Results written to {args.output_file}")

    # Print results to console
    print(f"Analysis Results (Bootstrap CI: {args.confidence * 100}%)\n")
    print("=" * 80 + "\n\n")

    for dir_name, result in results.items():
        print(f"Directory: {dir_name}\n")
        print(f"Number of samples: {result['n_samples']}\n")
        print(
            f"Accuracy: {result['mean_accuracy']:.3f} ({result['accuracy_ci'][0]:.2f}, {result['accuracy_ci'][1]:.2f})\n"
        )
        print(
            f"Inference time: {result['mean_inference_time']:.2f} ({result['mean_inference_time_low']:.2f}, {result['mean_inference_time_high']:.2f})\n"
        )
        print("\n" + "-" * 80 + "\n\n")

    baselm = "Base LM"
    direct = "TM LCD"
    lcd = "ARS LCD"
    twist = "Twisted SMC"
    swar = "AWRS SMC"
    sr = "Sample-verify"

    method2name = {
        "lm": baselm,
        "direct": direct,
        "lcd": lcd,
        "twist": twist,
        "swar": swar,
        "sr": sr,
    }

    def make_name(method, n_particles):
        if n_particles == 1:
            return f"{method2name[method]}"
        else:
            return f"{method2name[method]} (N = {n_particles})"

    plot_data = []
    for directory, result in results.items():
        method = directory.split("/")[-1].split("-")[0]  # Extract method name
        n_particles = int(directory.split("/")[-1].split("-")[-1])
        plot_data.append(
            {
                "method": make_name(method, n_particles),
                "m_name": method2name[method],
                "directory": directory,
                "accuracy": result["mean_accuracy"],
                "accuracy_low": result["accuracy_ci"][0],
                "accuracy_high": result["accuracy_ci"][1],
                "inference_time": result["mean_inference_time"],
                "inference_time_low": result["mean_inference_time_low"],
                "inference_time_high": result["mean_inference_time_high"],
                "samples": result["n_samples"],
                "time_per_sample": result["mean_inference_time"] / result["n_samples"],
            }
        )

    df = pd.DataFrame(plot_data)
    method_order = [baselm, direct, lcd, sr, twist, swar]
    df["m_name"] = pd.Categorical(df["m_name"], categories=method_order, ordered=True)

    y = "inference_time"
    x = "accuracy"

    x_min = df[f"{x}_low"].min() - 0.05
    x_max = df[f"{x}_high"].max() + 0.05
    y_min = df[f"{y}_low"].min() * 0.5  # For log scale, multiply by factor
    y_max = df[f"{y}_high"].max() * 2.0  # For log scale, multiply by factor

    # Create the plot
    plot = (
        ggplot(df, aes(x=x, y=y, color="m_name", label="method"))
        + geom_point(size=4, alpha=0.7)
        # Add horizontal error bars for accuracy confidence intervals
        + geom_errorbarh(
            aes(xmin=f"{x}_low", xmax=f"{x}_high", y=y), height=0.05, alpha=0.7
        )
        # Add vertical error bars for time confidence intervals
        + geom_errorbar(
            aes(ymin=f"{y}_low", ymax=f"{y}_high", x=x), width=0.005, alpha=0.7
        )
        # Offset labels slightly using position_nudge
        + geom_text(position=position_nudge(x=0.01, y=0.1), size=8, va="bottom")
        + labs(
            title="",
            x="Accuracy (95% CI)",
            y="Mean runtime (mins, log scale)",
            color="Method",
        )
        + scale_y_log10()
        + scale_color_brewer(type="qual", palette="Dark2")
        + coord_cartesian(
            xlim=(x_min, x_max), ylim=(y_min, y_max), expand=True
        )  # Set explicit limits with expansion
        + theme_light()
        + theme(
            plot_title=element_text(size=14, face="bold"),
            axis_title=element_text(size=12),
            legend_title=element_text(size=10),
            legend_position="bottom",
            legend_box="horizontal",
        )
    )

    # Save the plot
    plot_filename = "accuracy_vs_time.png"
    plot.save(plot_filename, dpi=300, width=6, height=6)
    print(f"Visualization saved as '{plot_filename}'")


if __name__ == "__main__":
    main()
