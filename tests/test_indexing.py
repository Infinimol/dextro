from dextro.indexing import FileIndexer, DirectoryIndexer, index_dataset
from .conftest import NUM_PARTITIONS, NUM_RECORDS_PER_PARTITION


def test_file_indexer(dataset_root):
    file_indexer = FileIndexer()

    records = list(file_indexer(dataset_root / "part_000.jsonl"))

    assert len(records) == NUM_RECORDS_PER_PARTITION


def test_directory_indexer(dataset_root):
    file_indexer = FileIndexer()
    directory_indexer = DirectoryIndexer(indexer=file_indexer)

    records = list(directory_indexer(dataset_root))

    assert len(records) == NUM_PARTITIONS * NUM_RECORDS_PER_PARTITION


def test_dataset_indexing(dataset_root):
    index_df = index_dataset(data_root=dataset_root, num_workers=1)

    assert len(index_df) == NUM_PARTITIONS * NUM_RECORDS_PER_PARTITION
    assert "filename" in index_df.columns
    assert "start" in index_df.columns
    assert "end" in index_df.columns
