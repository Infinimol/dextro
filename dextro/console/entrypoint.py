import argparse
from dextro.console import index_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Dextro: NLP Dataset Indexing for Blazing Fast Random Access"
    )
    subparsers = parser.add_subparsers(
        help="Command to execute", required=True, dest="command"
    )
    index_dataset_parser = subparsers.add_parser(
        "index-dataset", help="Create an index for a dataset."
    )
    index_dataset.add_arguments(index_dataset_parser)
    args = parser.parse_args()

    if args.command == "index-dataset":
        index_dataset.main(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
