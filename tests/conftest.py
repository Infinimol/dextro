import lorem
import tempfile
import json
import pytest

from pathlib import Path
from dextro.indexing import index_dataset

NUM_PARTITIONS = 3
NUM_RECORDS_PER_PARTITION = 5


def sample_record():
    return {"text": lorem.sentence()}


@pytest.fixture
def dataset_root():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        for partition_idx in range(NUM_PARTITIONS):
            records = [sample_record() for _ in range(NUM_RECORDS_PER_PARTITION)]
            partition_path = tmp_dir / f"part_{partition_idx:03d}.jsonl"
            partition_path.write_text(
                "\n".join(json.dumps(record) for record in records)
            )

        yield tmp_dir


@pytest.fixture
def record():
    return sample_record()


@pytest.fixture
def record_batch():
    return [sample_record() for _ in range(NUM_RECORDS_PER_PARTITION)]


@pytest.fixture
def indexed_dataset(dataset_root):
    index_df = index_dataset(dataset_root)
    index_df.write_parquet(dataset_root / 'index.parquet')
    yield dataset_root