from typing import Callable
import polars as pl
import mmap
import json
import atexit
from pathlib import Path
from torch.utils.data import Dataset


class IndexedDataset(Dataset):
    """
    PyTorch Dataset implementation for indexed datasets.

    Args:
        root: The root directory of the indexed dataset.
        index_filename: The filename of the index file. Defaults to 'index.parquet'.
        load_fn: A function to load the serialized item from the dataset. Defaults to json.loads.
        text_key: The key of the text field in the serialized item. Defaults to 'text'.
        index_filter: A filter in the form of a Polars expression to apply to the index. Defaults to None.
    """

    def __init__(
        self,
        root: str | Path,
        index_filename: str = "index.parquet",
        load_fn: Callable[[bytes], dict] = json.loads,
        text_key: str = "text",
        index_filter=None,
    ):
        self.root = Path(root)
        self.index = pl.read_parquet(self.root / index_filename)
        self.load_fn = load_fn
        self.text_key = text_key

        if index_filter is not None:
            self.index = self.index.filter(index_filter)

        self.filenames = self.index["filename"].unique().to_list()

        self.file_handles = {
            filename: (self.root / filename).open("r+b") for filename in self.filenames
        }

        self.mem_maps = {
            filename: mmap.mmap(
                self.file_handles[filename].fileno(), 0, access=mmap.ACCESS_READ
            )
            for filename in self.file_handles
        }

        atexit.register(self.cleanup)

    def cleanup(self):
        for filename in self.file_handles:
            self.file_handles[filename].close()
            self.mem_maps[filename].close()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        filename, start, end, *_ = self.index.row(idx)
        buffer = self.mem_maps[filename][start:end]
        item = self.load_fn(buffer)

        if isinstance(item, str):
            return item

        return item[self.text_key]
