import os
import sys
import json
import time
import pickle
import asyncio
import warnings
import argparse
from tqdm import tqdm
from pathlib import Path
from genlm_control import InferenceEngine, PromptedLLM
from genlm_control.potential.built_in.json import JsonSchema
from datasets import load_dataset


project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from benchmark.util import make_sampler  # noqa: E402

if "." not in sys.path:
    sys.path.insert(0, ".")

from util import few_shots_messages_formatter, evaluate_output  # noqa: E402

warnings.filterwarnings("once", category=RuntimeWarning)

os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "360"


def parse_args():
    parser = argparse.ArgumentParser(description="Run JSON schema evaluation.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--min_max_tokens",
        type=int,
        default=150,
        help="Minimum max tokens.",
    )
    parser.add_argument(
        "--n_particles",
        type=int,
        default=1,
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
        default="swar",
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
        help="Use the critic.",
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
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=[],
        help="Tasks to run.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Split to use.",
    )
    return parser.parse_args()


TASKS = [
    "Github_easy",
    "Github_hard",
    "Github_medium",
    "Github_trivial",
    "Github_ultra",
    "Github_very_hard",
    "Glaiveai2K",
    "JsonSchemaStore",
    "Kubernetes",
    "WashingtonPost",
    "Snowplow",
]


async def main():
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    arg_path = os.path.join(args.output_dir, "args.json")
    with open(arg_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    llm = PromptedLLM.from_name(args.model_name, **json.loads(args.lm_args))

    tasks = args.tasks if args.tasks else TASKS

    for task in tasks:
        print(f"Running inference for task: {task}")

        data = load_dataset("epfl-dlab/JSONSchemaBench", task)

        results_dir = os.path.join(args.output_dir, task)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        accs = []
        inference_times = []
        for i, schema_str in enumerate(tqdm(data[args.split]["json_schema"])):
            result_file = os.path.join(results_dir, f"{i}_result.pkl")

            if not args.overwrite and os.path.exists(result_file):
                print(
                    f"Skipping inference for schema {i} of task {task} (already exists)"
                )
                continue

            schema = json.loads(schema_str)
            potential = JsonSchema(schema)
            prompt_messages = few_shots_messages_formatter(task, schema)
            llm.prompt_ids = llm.model.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
            )

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

            max_tokens = max(
                int(len(llm.model.tokenizer.encode(schema_str)) * 1.5),
                args.min_max_tokens,
            )

            start_time = time.time()

            sequences = await InferenceEngine(sampler, critic=critic)(
                n_particles=args.n_particles,
                max_tokens=max_tokens,
                ess_threshold=args.ess_threshold,
                json_path=os.path.join(results_dir, f"{i}_record.json"),
                verbosity=args.verbosity,
            )

            this_inference_time = time.time() - start_time
            this_acc = evaluate_output(sequences, schema)
            accs.append(this_acc)
            inference_times.append(this_inference_time)

            print(f"Instance accuracy: {this_acc}")
            print(f"Overall accuracy: {sum(accs) / len(accs)}")
            print(f"Inference time: {this_inference_time}")
            print(
                f"Average inference time: {sum(inference_times) / len(inference_times)}"
            )

            metadata = {
                "schema": schema,
                "accuracy": this_acc,
                "model_name": args.model_name,
                "lm_args": args.lm_args,
                "max_tokens": max_tokens,
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
                    with open(os.path.join(results_dir, f"{i}_stats.pkl"), "wb") as f:
                        pickle.dump(stats, f)
                    sampler._reset_stats()
                else:
                    warnings.warn(
                        "Sampler does not support timing stats", RuntimeWarning
                    )

            if hasattr(sampler, "_save_cache"):
                sampler._save_cache()


if __name__ == "__main__":
    asyncio.run(main())
