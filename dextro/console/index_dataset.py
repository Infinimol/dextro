import argparse
import polars as pl

from pathlib import Path
from dextro.enrichers import enricher_registry
from dextro.loaders import loader_registry
from dextro.indexing import index_dataset


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "data_root",
        type=Path,
        help="The root directory of the dataset. It should contain dataset chunks in JSON Line format.",
    )
    parser.add_argument(
        "--output_filename",
        type=Path,
        default="index.parquet",
        help="Filename for the index file. Defaults to 'index.parquet'.",
    )
    parser.add_argument(
        "--glob",
        nargs="*",
        default=["*.jsonl", "*.json"],
        help="Glob patterns to match dataset chunk files. Defaults to ['*.jsonl', '*.json']. "
        "Note that .json files will still be interpreted as JSON Line files.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        help="Maximum number of records to index. Useful for testing. Defaults to None (index all records).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Internal processing batch size. Higher values may improve the efficiency of batched enrichers.",
    )
    parser.add_argument(
        "--loader",
        choices=list(loader_registry.keys()),
        default="jsonl",
        help="The loader to use for reading the dataset. Defaults to 'jsonl'.",
    )
    parser.add_argument(
        "--enrichers",
        nargs="*",
        choices=list(enricher_registry.keys()),
        default=[],
        help="Enrichers to use for indexing. By default, no enrichers will be used.",
    )
    parser.add_argument(
        "--categorical_columns",
        nargs="*",
        default=["filename"],
        help="Columns to cast as categorical. Defaults to ['filename'].",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress bar.")

    return parser


def main(args: argparse.Namespace):
    root = args.data_root

    enrichers = [enricher_registry[name]() for name in args.enrichers]
    loader = loader_registry[args.loader]()
    
    index_df = index_dataset(
        data_root=root,
        batch_size=args.batch_size,
        loader=loader,
        max_iter=args.max_iter,
        glob=args.glob,
        enrichers=enrichers,
        progress_bar=not args.quiet,
    )

    for col in set(index_df.columns) & set(args.categorical_columns):
        index_df = index_df.with_columns(pl.col(col).cast(pl.Categorical))

    index_df.write_parquet(root / args.output_filename)
