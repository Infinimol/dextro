from .conftest import NUM_PARTITIONS, NUM_RECORDS_PER_PARTITION
from dextro.dataset import IndexedDataset


def test_indexed_dataset(indexed_dataset):
    dataset = IndexedDataset(indexed_dataset)

    assert len(dataset) == NUM_PARTITIONS * NUM_RECORDS_PER_PARTITION

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert 'text' in item
