import os
import sys
import asyncio
from tqdm import tqdm
from pathlib import Path
from genlm_control import BoolCFG

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from benchmark.text_to_sql.spider.dialogue import load_spider_data  # noqa: E402


async def validate(raw_spider_dir, grammar_dir):
    """
    Validate that all queries in the Spider dataset can be parsed by their respective grammars.

    Args:
        raw_spider_dir: Path to the Spider dataset directory
        grammar_dir: Path to the directory containing grammar files

    Returns:
        list: Failed queries that couldn't be parsed
    """
    # Load Spider data
    raw_spider_dir = Path(raw_spider_dir)
    dev_data = load_spider_data(raw_spider_dir / "dev.json")

    # Cache for grammars to avoid reloading
    cfg_cache = {}

    complete_failed = []
    prefix_failed = []

    # Validate each query against its grammar
    for datum in tqdm(dev_data):
        bool_cfg = cfg_cache.get(datum.schema_name)
        if bool_cfg is None:
            grammar_path = os.path.join(grammar_dir, f"{datum.schema_name}.lark")
            with open(grammar_path, "r") as f:
                grammar = f.read()
            bool_cfg = BoolCFG.from_lark(grammar)
            cfg_cache[datum.schema_name] = bool_cfg

        w = await bool_cfg.complete(datum.query.encode("utf-8"))
        if w != 0:
            complete_failed.append((datum.schema_name, datum.query))

        w = await bool_cfg.prefix(datum.query.encode("utf-8"))
        if w != 0:
            prefix_failed.append((datum.schema_name, datum.query))

    return complete_failed, prefix_failed


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate SQL grammars against Spider dataset"
    )
    parser.add_argument(
        "--spider-dir",
        type=str,
        default="data/spider_data",
        help="Path to the Spider dataset directory",
    )
    parser.add_argument(
        "--grammar-dir",
        type=str,
        default="data/grammars",
        help="Path to the directory containing grammar files",
    )
    args = parser.parse_args()

    complete_failed, prefix_failed = await validate(args.spider_dir, args.grammar_dir)

    if not complete_failed and not prefix_failed:
        print("All queries parsed successfully!")
    else:
        if not complete_failed:
            print(f"Failed to parse {len(complete_failed)} queries:")
            for schema_name, query in complete_failed:
                print(f"Schema: {schema_name}, Query: {query}")
        if not prefix_failed:
            print(f"Failed to parse {len(prefix_failed)} queries as prefixes:")
            for schema_name, query in prefix_failed:
                print(f"Schema: {schema_name}, Query: {query}")


if __name__ == "__main__":
    asyncio.run(main())
