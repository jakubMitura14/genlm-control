import os
import sys
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from arsenal.maths import logsumexp
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed

from genlm_control.constant import EndOfSequence

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from benchmark.text_to_sql.spider.evaluator import Evaluator  # noqa: E402


@lru_cache
def cached_eval(x, y, db):
    return evaluator.evaluate(x, y, db_name=db)


def initialize_worker(raw_spider_dir, timeout=None):
    global evaluator
    evaluator = Evaluator(raw_spider_dir, timeout=timeout)


def posterior_weighted_eval(contexts, log_weights, gold, db, utterance):
    weighted_acc = 0
    particle_results = []

    log_total = logsumexp(log_weights)
    log_normalized_weights = log_weights - log_total
    probs = np.exp(log_normalized_weights)

    assert len(contexts) == len(probs)

    for context, p in zip(contexts, probs):
        ctx = context[:-1] if isinstance(context[-1], EndOfSequence) else context

        try:
            pred = b"".join(ctx).decode("utf-8")
        except UnicodeDecodeError:
            w_acc = 0
            particle_results.append((context, w_acc))
            continue

        if np.isnan(p):
            w_acc = 0
        else:
            acc = cached_eval(gold, pred, db)
            w_acc = p * acc[0]

        weighted_acc += w_acc
        particle_results.append((context, w_acc))

    return {
        "result": weighted_acc,
        "particle_results": particle_results,
        "gold": gold,
        "db": db,
        "utterance": utterance,
    }


def eval_wrapper(datum):
    return posterior_weighted_eval(
        datum["contexts"],
        datum["log_weights"],
        datum["metadata"]["gold"],
        datum["metadata"]["schema_name"],
        datum["metadata"]["utterance"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help='Path to the directory containing the results. Each file in this directory should be named "i_result.pkl", where i is an integer representing the index of the result.',
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=2,
        help="Number of workers to use for evaluation",
    )
    parser.add_argument(
        "--raw_spider_dir",
        type=str,
        default="data/spider_data",
        help="Path to the raw Spider dataset",
    )
    parser.add_argument(
        "--output_pkl",
        type=str,
        required=True,
        help="Path to save the evaluation results in pickled format.",
    )
    parser.add_argument(
        "--timeout", type=int, default=None, help="Timeout for evaluation"
    )
    args = parser.parse_args()

    data = []
    for file in os.listdir(args.results_dir):
        if file.endswith("_result.pkl"):
            with open(os.path.join(args.results_dir, file), "rb") as f:
                data.append(pickle.load(f))

    raw_spider_dir = Path(args.raw_spider_dir)
    if not raw_spider_dir.exists():
        raise FileNotFoundError(
            f"Raw Spider dataset directory not found: {raw_spider_dir}"
        )

    print(
        "Total inference time (min): ",
        sum([datum["metadata"]["inference_time"] for datum in data]) / 60,
    )

    if args.n_workers > 1:
        with ProcessPoolExecutor(
            initializer=initialize_worker,
            initargs=(raw_spider_dir, args.timeout),
            max_workers=args.n_workers,
        ) as executor:
            with tqdm(total=len(data)) as progress_bar:
                futures = [executor.submit(eval_wrapper, datum) for datum in data]
                results = []
                for future in as_completed(futures):
                    results.append(future.result())
                    progress_bar.update(1)
    else:
        initialize_worker(raw_spider_dir, args.timeout)
        results = [eval_wrapper(datum) for datum in tqdm(data)]

    mean, lower, upper = mean_ci([r["result"] for r in results])
    print(f"Mean accuracy: {round(mean, 4)} ({round(lower, 2)}, {round(upper, 2)})")

    with open(args.output_pkl, "wb") as f:
        pickle.dump(results, f)


def mean_ci(values, ci=0.95, n_bootstrap=10000):
    mean = np.mean(values)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    lower_percentile = (1 - ci) / 2 * 100
    upper_percentile = (1 + ci) / 2 * 100
    lower, upper = np.percentile(bootstrap_means, [lower_percentile, upper_percentile])
    return mean, lower, upper


if __name__ == "__main__":
    main()
