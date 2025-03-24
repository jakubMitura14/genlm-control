import os
import sys
import numpy as np
import pandas as pd
import json
import time
import pickle
import asyncio
import warnings
import argparse
import string
import regex
from tqdm import tqdm
from pathlib import Path
from genlm_control import InferenceEngine, PromptedLLM, Potential
from genlm_control.constant import EndOfSequence

# import multiprocessing # For vllm tensor parallelism.
# multiprocessing.set_start_method('spawn', force=True)

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from benchmark.util import make_sampler  # noqa: E402

warnings.filterwarnings("once", category=RuntimeWarning)

os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "360"


class PatternPotential(Potential):
    def __init__(self, pattern):
        vocab = list(map(ord, string.printable))
        super().__init__(vocab)
        self.r = regex.compile(pattern)

    async def complete(self, context):
        text = "".join(map(chr, context))
        match = self.r.fullmatch(text) is not None
        return 0.0 if match else float("-inf")

    async def prefix(self, context):
        text = "".join(map(chr, context))
        match = self.r.match(text, partial=True) is not None
        return 0.0 if match else float("-inf")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Spider inference with table column check potential."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=100, help="Maximum number of tokens."
    )
    parser.add_argument(
        "--n_particles",
        type=int,
        default=10,
        help="Number of particles.",
    )
    parser.add_argument(
        "--ess_threshold",
        type=float,
        default=0.9,
        help="ESS threshold for resampling.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write the inference results.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing inference results in the output directory.",
    )
    parser.add_argument(
        "--sampler_name",
        type=str,
        default="eager",
        help="Name of the sampler to use.",
    )
    parser.add_argument(
        "--sampler_args",
        type=str,
        default="{}",
        help="Arguments for the sampler.",
    )
    parser.add_argument(
        "--time_sampler",
        action="store_true",
        help="Time the sampler operations.",
    )
    parser.add_argument(
        "--use_critic",
        action="store_true",
        help="Use the critic to check the table column references.",
    )
    parser.add_argument(
        "--lm_args",
        type=str,
        default="{}",
        help="Arguments to pass to the language model, provided to PromptedLLM at initialization.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=0,
        help="Verbosity level for inference. When set to 1, particles are printed at each step.",
    )

    return parser.parse_args()


prompt_template = """
Generate a simple string that matches the specified pattern.

Here are some examples:

Regex: (ab)+
String: ab

Regex: (ab|cd)+
String: cd

It is important that you only output the string matching the regex and nothing else.

Regex: {}
String:"""


async def main():
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    arg_path = os.path.join(args.output_dir, "args.json")
    with open(arg_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    llm = PromptedLLM.from_name(args.model_name, **json.loads(args.lm_args))
    eos = [t for t in llm.vocab if b"\n" in t]
    llm = llm.spawn_new_eos(eos)

    print("\n------ Loaded LM ------\n")

    data = pd.read_csv(
        "/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/regex/benchmark.csv"
    )["regex"].to_list()

    accs = []
    inference_times = []
    for i, pattern in enumerate(tqdm(data, desc="Running inference")):
        result_file = os.path.join(args.output_dir, f"{i}_result.pkl")

        if not args.overwrite and os.path.exists(result_file):
            print(f"Skipping inference for pattern: {pattern} (already exists)")
            continue

        print(f"Running inference for pattern: {pattern}")

        potential = PatternPotential(pattern)

        llm.set_prompt_from_str(prompt_template.format(pattern))

        sampler = make_sampler(
            args.sampler_name,
            llm,
            potential,
            json.loads(args.sampler_args),
            args.time_sampler,
        )

        critic = None
        if args.use_critic:
            critic = potential.coerce(llm, f=b"".join)

        start_time = time.time()

        sequences = await InferenceEngine(sampler, critic=critic)(
            n_particles=args.n_particles,
            max_tokens=args.max_tokens,
            ess_threshold=args.ess_threshold,
            json_path=os.path.join(args.output_dir, f"{i}_record.json"),
            verbosity=args.verbosity,
        )

        this_inference_time = time.time() - start_time

        this_acc = 0
        for sequence, p in sequences.posterior.items():
            if np.isnan(p):
                continue
            if isinstance(sequence[-1], EndOfSequence):
                sequence = sequence[:-1]
            else:
                continue  # If the sequence does not end with an EOS token, it is not a valid output.
            try:
                text = b"".join(sequence).decode("utf-8")
                this_acc += (
                    p if regex.compile(pattern).fullmatch(text) is not None else 0
                )
            except UnicodeDecodeError:
                continue  # If the sequence contains non-UTF-8 characters, skip it, as it will not be a valid output.

        accs.append(this_acc)
        inference_times.append(this_inference_time)

        print(f"Instance accuracy: {this_acc}")
        print(f"Overall accuracy: {sum(accs) / len(accs)}")
        print(f"Inference time: {this_inference_time}")
        print(f"Average inference time: {sum(inference_times) / len(inference_times)}")

        metadata = {
            "pattern": pattern,
            "accuracy": this_acc,
            "model_name": args.model_name,
            "lm_args": args.lm_args,
            "max_tokens": args.max_tokens,
            "sampler_name": args.sampler_name,
            "sampler_args": args.sampler_args,
            "inference_time": this_inference_time,
        }

        with open(result_file, "wb") as f:
            pickle.dump(
                {
                    "metadata": metadata,
                    "contexts": sequences.contexts,
                    "log_weights": sequences.log_weights,
                },
                f,
            )

        # Save proposal timing stats.
        if args.time_sampler:
            if hasattr(sampler, "get_stats"):
                stats = sampler.get_stats()
                stats["metadata"] = metadata
                with open(os.path.join(args.output_dir, f"{i}_stats.pkl"), "wb") as f:
                    pickle.dump(stats, f)
                sampler._reset_stats()
            else:
                warnings.warn("Sampler does not support timing stats", RuntimeWarning)

        if hasattr(sampler, "_save_cache"):
            sampler._save_cache()


if __name__ == "__main__":
    asyncio.run(main())
