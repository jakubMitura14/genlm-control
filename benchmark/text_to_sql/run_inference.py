import os
import sys
import json
import pickle
import asyncio
import argparse
from tqdm import tqdm
from pathlib import Path

from genlm_control import InferenceEngine, PromptedLLM, BoolCFG

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from benchmark.util import TimedTokenSampler  # noqa: E402
# from benchmark.text_to_sql.potential import SpiderTableColumnVerifier  # noqa: E402

from benchmark.text_to_sql.spider.schema import load_schemas  # noqa: E402
from benchmark.text_to_sql.spider.dialogue import load_spider_data  # noqa: E402
from benchmark.text_to_sql.spider.prompt_formatter import SpiderPromptFormatter  # noqa: E402


def get_sampler(sampler_name):
    if sampler_name == "eager":
        from genlm_control.sampler import eager_token_sampler

        return eager_token_sampler
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")


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

    sampler_cache = {}
    # critic_cache = {}
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
        if not args.overwrite and os.path.exists(
            os.path.join(args.output_dir, f"{i}_result.json")
        ):
            pbar.set_postfix(status=f"Skipped {i} (exists)")
            continue

        pbar.set_postfix(status=f"{i}")

        llm.prompt_ids = llm.model.tokenizer.apply_chat_template(
            prompt_formatter.format_openai(datum),
            add_generation_prompt=True,
            tokenize=True,
        )

        # Load the grammar.
        grammar = open(
            os.path.join(args.grammar_dir, f"{datum.schema_name}.lark"), "r"
        ).read()

        # Fetch or create the sampler.
        sampler_key = (grammar, args.sampler_name, args.sampler_args)
        if sampler_key not in sampler_cache:
            bool_cfg = BoolCFG.from_lark(grammar)
            sampler_cache[sampler_key] = get_sampler(args.sampler_name)(
                llm, bool_cfg, **json.loads(args.sampler_args)
            )
        sampler = sampler_cache[sampler_key]

        # Fetch or create the critic.
        if args.use_critic:
            # critic_key = (grammar, datum.schema_name)
            # if critic_key not in critic_cache:
            #    critic_cache[critic_key] = SpiderTableColumnVerifier(
            #        grammar, spider_schemas[datum.schema_name]
            #    )
            # critic = critic_cache[critic_key]
            raise NotImplementedError("Critic not implemented")
        else:
            critic = None

        if args.time_sampler:
            sampler = TimedTokenSampler(sampler)

        # Run engine with record saving.
        sequences = await InferenceEngine(sampler, critic=critic)(
            n_particles=args.n_particles,
            max_tokens=args.max_tokens,
            ess_threshold=args.ess_threshold,
            json_path=os.path.join(args.output_dir, f"{i}_record.json"),
        )

        metadata = {
            "gold": datum.query,
            "utterance": datum.utterance,
            "schema_name": datum.schema_name,
            "model_name": args.model_name,
            "lm_args": args.lm_args,
            "max_tokens": args.max_tokens,
            "sampler_name": args.sampler_name,
            "sampler_args": args.sampler_args,
        }

        with open(os.path.join(args.output_dir, f"{i}_result.pkl"), "wb") as f:
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
            assert len(sampler.sample_times) <= args.n_particles * args.max_tokens
            timing_stats = sampler.get_timing_stats()
            timing_stats["metadata"] = metadata
            with open(os.path.join(args.output_dir, f"{i}_timing.json"), "w") as f:
                json.dump(timing_stats, f)


if __name__ == "__main__":
    asyncio.run(main())
