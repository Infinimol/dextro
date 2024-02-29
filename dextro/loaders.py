import abc
import json
import csv
from pathlib import Path
import polars as pl
from typing import Iterable
from dextro.types import DatasetRecord, PathLike, FileItem, ItemMeta


class BaseLoader(abc.ABC):
    @abc.abstractmethod
    def load_item(self, content: bytes) -> DatasetRecord:
        pass

    @abc.abstractmethod
    def iter_file_items(self, path: PathLike) -> Iterable[FileItem]:
        pass


class JSONLinesLoader(BaseLoader):
    def load_item(self, content: bytes):
        return json.loads(content)
    
    def iter_file_items(self, path: PathLike) -> Iterable[FileItem]:
        path = Path(path)
        
        with open(path, 'rb') as fp:
            while True:
                start = fp.tell()

                line = fp.readline().strip()

                if not line:
                    break

                end = start + len(line)

                data = self.load_item(line)

                yield FileItem(
                    meta=ItemMeta(start=start, end=end, filename=path.name),
                    data=data
                )


class CSVLoader(BaseLoader):
    def __init__(self, header: int | None = 0, names: list[str] | None = None, delimiter: str = ',', encoding: str = 'utf8'):
        self.header = header
        self.names = names
        self.delimiter = delimiter
        self.encoding = encoding
    
    def load_item(self, content: bytes) -> DatasetRecord:
        row, *_ =  csv.reader([
            content.decode(self.encoding)
        ], delimiter=self.delimiter)
        return row

    def iter_file_items(self, path: PathLike) -> Iterable[FileItem]:
        path = Path(path)

        names = self.names
        line_idx = 0

        with open(path, 'rb') as fp:
            while True:
                start = fp.tell()

                line = fp.readline().strip()

                if not line:
                    break

                end = start + len(line)

                if self.header is not None and names is None:
                    if line_idx < self.header:
                        continue
                    else:
                        names = self.load_item(line)
                    
                    start = end + 1

                    continue

                row = self.load_item(line)

                if names is None:
                    names = [f'col_{i}' for i in range(len(row))]
                else:
                    assert len(names) == len(row)
                
                record = dict(zip(names, row, strict=True))

                yield FileItem(
                    meta=ItemMeta(start=start, end=end, filename=path.name),
                    data=record
                )

                start = end + 1


default_loader = JSONLinesLoader()

loader_registry = {
    'jsonl': lambda: JSONLinesLoader(),
    'json_lines': lambda: JSONLinesLoader(),
    'csv': lambda: CSVLoader(delimiter=','),
    'csv_space_seperated': lambda: CSVLoader(delimiter=' '),
    'tsv': lambda: CSVLoader(delimiter='\t'),
}
