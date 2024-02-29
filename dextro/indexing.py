from itertools import islice
import json
import warnings
import polars as pl
import multiprocessing as mp

from tqdm import tqdm
from pathlib import Path
from typing import Callable, Any, Iterator, Sequence
from dextro.enrichers import BaseEnricher, BaseBatchedEnricher, Enricher
from dextro.types import DatasetRecord, FileItem, PathLike
from dextro.loaders import BaseLoader, default_loader

EnricherFunction = Callable[[DatasetRecord], DatasetRecord]
BatchedEnricherFunction = Callable[[list[DatasetRecord]], list[DatasetRecord]]


class FileIndexer:
    """
    Indexes records in a single file.

    Args:
        open_fn: A function to open the file. Defaults to open.
        load_fn: A function to load the serialized item from the file. Defaults to json.loads.
        batch_size: Internal processing batch size. Higher values may improve the efficiency of batched enrichers. Defaults to 1.
        enrichers: Enrichers to use for indexing. By default, no enrichers will be used.
    """

    KEEP_KEYS = ["filename", "start", "end", "text_length"]

    def __init__(
        self,
        loader: BaseLoader = default_loader,
        batch_size: int = 1,
        enrichers: Enricher | Sequence[Enricher] | None = None,
    ):
        self.loader = loader
        self.batch_size = batch_size
        self.enrichers = []
        self.batched_enrichers = []

        if enrichers is None:
            enrichers = []
        elif isinstance(enrichers, Enricher):
            enrichers = [enrichers]

        for enricher in enrichers:
            if isinstance(enricher, Enricher):
                if isinstance(enricher, BaseEnricher):
                    self.enrichers.append(enricher)
                if isinstance(enricher, BaseBatchedEnricher):
                    self.batched_enrichers.append(enricher)
            else:
                raise ValueError(f"Expected Enricher, got {enricher!r}")

        if self.batched_enrichers:
            if self.batch_size == 1:
                warnings.warn(
                    "Batched transform functions provided, but batch size is 1. "
                    "This might lead to inefficient batched processing."
                )

    def _enrich_item(self, item: DatasetRecord):
        for transform in self.enrichers:
            item = transform.enrich_item(item)
            if item is None:
                break
        return item

    def _enrich_batch(self, batch: list[DatasetRecord]):
        for transform in self.batched_enrichers:
            batch = transform.enrich_batch(batch)
            if batch is None:
                break
            batch = [item for item in batch if item is not None]
            if not batch:
                break
        return batch
    
    def _finish_batch(self, batch: list[FileItem]):
        for item in batch:
            yield item.meta.as_dict()

    def __call__(self, path: PathLike) -> Iterator[DatasetRecord]:
        """
        Indexes records in a single file.

        The function will iterat over the lines of the given file, apply enrichers and yield the resulting records.

        Processing happens in chunks of `batch_size` as provided in the constructor. This is useful for enrichers that
        can profit from batched processing, such as neural networks.

        Args:
            path: The path to the file to index.

        Yields:
            A dictionary representing a single record in the dataset.
            The keys 'filename', 'start', 'end', and 'text_length' are always present while
            additional keys prefixed with 'meta_' may be present depending on the enrichers used.
        """
        path = Path(path)

        batch = []

        for item in self.loader.iter_file_items(path):
            item = self._enrich_item(item)

            if not item:
                continue

            batch.append(item)

            if len(batch) >= self.batch_size:
                batch = self._enrich_batch(batch)
                yield from self._finish_batch(batch)

                batch = []

        if batch:
            batch = self._enrich_batch(batch)
            yield from self._finish_batch(batch)


class DirectoryIndexer:
    """
    Indexes dataset partititions in a directory.

    Args:
        indexer: The indexer to use for indexing individual files.
        glob: Glob patterns to match dataset chunk files. Defaults to ['*.jsonl', '*.json'].
        num_workers: The number of worker processes to use for indexing. Defaults to None (no parallelism).
            It is recommended to not use parallelization at the moment as the overhead may lead
            to a significant overhead and thus slower indexing than single-threaded processing.
    """

    def __init__(
        self,
        indexer: FileIndexer,
        glob: str | list[str] = ("*.jsonl", "*.json"),
        num_workers: int | None = None,
    ):
        if isinstance(glob, str):
            glob = [glob]

        self.indexer = indexer
        self.glob = glob
        self.num_workers = num_workers

    def _collect_items(self, path, queue):
        for item in self.indexer(path):
            queue.put(item)
        queue.put(None)

    def __call__(self, dataset_root: Path | str) -> Iterator[DatasetRecord]:
        """
        Indexes dataset partititions in a directory.

        The function will iterate over the files matching the glob patterns in the given directory,
        apply the indexer to each file and yield the resulting records.

        Args:
            dataset_root: The root directory of the dataset.


        Yields:
            A dictionary representing a single record in the dataset.
            The keys 'filename', 'start', 'end', and 'text_length' are always present while
            additional keys prefixed with 'meta_' may be present depending on the enrichers used.
        """
        dataset_root = Path(dataset_root)

        paths = sorted(path for pattern in self.glob for path in dataset_root.glob(pattern))

        if not paths:
            raise ValueError(f"No files found matching glob patterns {self.glob!r} in {dataset_root}")

        if self.num_workers:
            with mp.Pool(self.num_workers) as pool:
                manager = mp.Manager()
                queue = manager.Queue()

                futures = [
                    pool.apply_async(
                        self._collect_items, kwds=dict(path=path, queue=queue)
                    )
                    for path in paths
                ]

                remaining = len(paths)

                while remaining > 0:
                    item = queue.get()

                    if item is None:
                        remaining -= 1
                    else:
                        yield item

                for future in futures:
                    future.get()
        else:
            for path in paths:
                yield from self.indexer(path)


def index_dataset(
    data_root: Path,
    batch_size: int = 1,
    max_iter: int | None = None,
    loader: BaseLoader = default_loader,
    glob: str | list[str] = ("*.jsonl", "*.json"),
    num_workers: int | None = None,
    enrichers: list[Enricher] | None = None,
    progress_bar: bool = True,
):
    """
    Indexes a dataset and returns the index as Polars DataFrame.

    Args:
        data_root: The root directory of the dataset. It should contain dataset chunks in JSON Line format.
        text_key: The key of the text field in the serialized item. Defaults to 'text'.
        batch_size: Internal processing batch size. Higher values may improve the efficiency of batched enrichers. Defaults to 1.
        max_iter: Maximum number of records to index. Useful for testing. Defaults to None (index all records).
        glob: Glob patterns to match dataset chunk files. Defaults to ['*.jsonl', '*.json'].
        num_workers: The number of worker processes to use for indexing. Defaults to None (no parallelism).
            It is recommended to not use parallelization at the moment as the overhead may lead
            to a significant overhead and thus slower indexing than single-threaded processing.
        enrichers: Enrichers to use for indexing. By default, no enrichers will be used.
        progress_bar: Whether to display a progress bar. Defaults to True.

    Returns:
        A Polars DataFrame representing the index of the dataset.
    """
    data_root = Path(data_root)

    file_indexer = FileIndexer(
        loader=loader,
        batch_size=batch_size,
        enrichers=enrichers
    )

    directory_indexer = DirectoryIndexer(
        indexer=file_indexer, glob=glob, num_workers=num_workers
    )

    records = directory_indexer(data_root)

    if max_iter:
        records = islice(records, max_iter)

    if progress_bar:
        records = tqdm(records)

    return pl.from_dicts(list(records))
