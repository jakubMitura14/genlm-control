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

from genlm_control import InferenceEngine, PromptedLLM, BoolCFG

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# from benchmark.text_to_sql.potential import SpiderTableColumnVerifier  # noqa: E402

from benchmark.util import make_sampler  # noqa: E402
from benchmark.text_to_sql.spider.schema import load_schemas  # noqa: E402
from benchmark.text_to_sql.spider.dialogue import load_spider_data  # noqa: E402
from benchmark.text_to_sql.spider.prompt_formatter import SpiderPromptFormatter  # noqa: E402

warnings.filterwarnings("once", category=RuntimeWarning)

os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "360"


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
        "--raw_spider_dir",
        type=str,
        default="data/spider_data",
        help="Path to the raw Spider data directory.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=100, help="Maximum number of tokens."
    )
    parser.add_argument(
        "--grammar_dir",
        type=str,
        default="data/grammars",
        help="Path to the Spider grammar directory.",
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
        "--dev_data_min",
        type=int,
        default=0,
        help="Limit on the number of dev data points to process.",
    )
    parser.add_argument(
        "--dev_data_max",
        type=int,
        default=1034,
        help="Limit on the number of dev data points to process.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
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
        "--cache_clear_interval",
        type=int,
        default=100,
        help="Interval to clear the parser caches.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=0,
        help="Verbosity level for inference. When set to 1, particles are printed at each step.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout for the inference engine.",
    )

    return parser.parse_args()


def spider_setup(raw_spider_dir):
    raw_spider_dir = Path(raw_spider_dir)
    dev_data = load_spider_data(raw_spider_dir / "dev.json")
    spider_schemas = load_schemas(
        schemas_path=raw_spider_dir / "tables.json", db_path=raw_spider_dir / "database"
    )
    train_data = load_spider_data(raw_spider_dir / "train_spider.json")
    prompt_formatter = SpiderPromptFormatter(train_data, spider_schemas)
    return dev_data, spider_schemas, prompt_formatter


async def main():
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    arg_path = os.path.join(args.output_dir, "args.json")
    with open(arg_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    dev_data, _, prompt_formatter = spider_setup(args.raw_spider_dir)

    cfg_cache = {}
    sampler_cache = {}
    critic_cache = {}
    llm = PromptedLLM.from_name(args.model_name, **json.loads(args.lm_args))

    filtered_dev_data = [
        (i, datum)
        for i, datum in enumerate(dev_data)
        if args.dev_data_min <= i < args.dev_data_max
    ]

    pbar = tqdm(
        filtered_dev_data, total=len(filtered_dev_data), desc="Running inference"
    )

    for i, datum in pbar:
        result_file = os.path.join(args.output_dir, f"{i}_result.pkl")

        if not args.overwrite and os.path.exists(result_file):
            pbar.set_postfix(status=f"Skipped {i} (exists)")
            continue

        pbar.set_postfix(status=f"{i}")

        if (i + 1) % args.cache_clear_interval == 0:
            for bool_cfg in cfg_cache.values():
                bool_cfg.clear_cache()

        llm.prompt_ids = llm.model.tokenizer.apply_chat_template(
            prompt_formatter.format_openai(datum),
            add_generation_prompt=True,
            tokenize=True,
        )

        grammar = open(
            os.path.join(args.grammar_dir, f"{datum.schema_name}.lark"), "r"
        ).read()

        cfg_key = (grammar, datum.schema_name)
        sampler_key = (grammar, args.sampler_name, args.sampler_args)
        critic_key = (grammar, datum.schema_name)

        if cfg_key not in cfg_cache:
            cfg_cache[cfg_key] = BoolCFG.from_lark(grammar)
        bool_cfg = cfg_cache[cfg_key]

        if sampler_key not in sampler_cache:
            sampler_cache[sampler_key] = make_sampler(
                args.sampler_name,
                llm,
                bool_cfg,
                json.loads(args.sampler_args),
                args.time_sampler,
            )
        sampler = sampler_cache[sampler_key]

        critic = None
        if args.use_critic:
            if critic_key not in critic_cache:
                critic_cache[critic_key] = bool_cfg.coerce(llm, f=b"".join)
            critic = critic_cache[critic_key]

        start_time = time.time()

        try:
            sequences = await asyncio.wait_for(
                InferenceEngine(sampler, critic=critic)(
                    n_particles=args.n_particles,
                    max_tokens=args.max_tokens,
                    ess_threshold=args.ess_threshold,
                    json_path=os.path.join(args.output_dir, f"{i}_record.json"),
                    verbosity=args.verbosity,
                ),
                timeout=args.timeout,
            )
        except asyncio.TimeoutError:
            print(f"Inference for example {i} timed out after {args.timeout} seconds")
            sequences = None

        metadata = {
            "gold": datum.query,
            "utterance": datum.utterance,
            "schema_name": datum.schema_name,
            "model_name": args.model_name,
            "lm_args": args.lm_args,
            "max_tokens": args.max_tokens,
            "sampler_name": args.sampler_name,
            "sampler_args": args.sampler_args,
            "inference_time": time.time() - start_time,
        }

        with open(result_file, "wb") as f:
            pickle.dump(
                {
                    "metadata": metadata,
                    "contexts": sequences.contexts if sequences else None,
                    "log_weights": sequences.log_weights if sequences else None,
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
