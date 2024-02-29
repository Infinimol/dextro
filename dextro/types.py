from dataclasses import dataclass, field
from typing import Any
from pathlib import Path


DatasetRecord = dict[str, Any]
PathLike = str | Path


@dataclass
class ItemMeta:
    start: int
    end: int
    filename: str
    additional_info: dict[str, Any] = field(default_factory=dict)

    def as_dict(self, meta_prefix: str = "meta_"):
        return {
            "start": self.start,
            "end": self.end,
            "filename": self.filename,
            **{meta_prefix + k: v for k, v in self.additional_info.items()}
        }


@dataclass
class FileItem:
    meta: ItemMeta
    data: DatasetRecord

    def add_info(self, key: str, value: Any):
        self.meta.additional_info[key] = value
