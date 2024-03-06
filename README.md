# Dextro: Dataset Indexing for Blazing Fast Random Access

**Dextro** is a streamlined indexing toolkit designed for large, multi-file text datasets. It enables O(1) random access to any dataset sample through memory mapping, eliminating the need for preloading. This toolkit is essential for researchers and developers working with extensive language datasets, offering a significant leap in processing and training flexibility without altering the original data format.

## Motivation

The ongoing revolution in artificial intelligence, particularly in LLM, is heavily reliant on extensive language datasets. However, these datasets often come in simple, non-indexed formats like JSON Lines, posing challenges for data handling. These challenges include the need for loading entire datasets into RAM for quick access, the limitations of sequential streaming, and the constraints on processing and training flexibility due to non-indexed formats.

Dextro addresses these challenges by enabling the efficient indexing of large, multi-file datasets without altering the original data. The index tracks the start and end positions of each sample within its source file, along with optional metadata for enhanced filtering capabilities. Through memory mapping, Dextro achieves O(1) random access to any record across multiple files, significantly improving data handling efficiency.

## Getting Started

### Installation

Install Dextro easily via pip:

```bash
pip install dextro
```

Install with all dependencies:

```bash
pip install dextro[all]
```

### Index Your Dataset

Dextro works with datasets in JSON Lines format, split across multiple files. To index such a dataset, organize your files as follows:

```
dataset/
    part001.jsonl
    part002.jsonl
    ...
    part999.jsonl
```

Example content (`dataset/part001.jsonl`):
```json
{"text": "first item", ...}
{"text": "second item", ...}
```

Run the following command to index your dataset, creating an `index.parquet` file in the dataset folder:

```bash
dextro index-dataset dataset/
```

This index file includes the filename, start, and end positions for each sample, facilitating efficient data access.

### Accessing Indexed Datasets

Dextro integrates with PyTorch's `Dataset` class, allowing for easy loading of indexed datasets. Here's how to sequentially iterate through your dataset:

```python
from tqdm import tqdm
from dextro.torch import IndexedDataset

dataset = IndexedDataset(data_root='dataset/')

for text in tqdm(dataset):
    pass
```

To demonstrate random access with shuffling, you can use a `DataLoader` as follows:

```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=128, shuffle=True)

for batch in tqdm(loader):
    pass
```

Dextro's memory mapping ensures that only the accessed data is loaded into memory, optimizing resource usage.

## Performance

Thanks to its minimal overhead and efficient data access, Dextro can process large NLP datasets at speeds close to those of reading directly from SSDs. This capability makes it possible to navigate through terabytes of data within minutes, even on consumer-grade storage.

## Comparison to 🤗 Datasets

The [🤗 Datasets](https://huggingface.co/docs/datasets) library also features [memory-mapped loading of partitioned datasets](https://huggingface.co/learn/nlp-course/en/chapter5/4). However, as of February 2024, it lacks the capability for random access, and shuffled iteration across a dataset is confined to the limits of an item buffer. Moreover, 🤗 Datasets does not offer the functionality to pre-filter data through a lightweight dataset index.

## Advanced Features

### Index Enrichers

Dextro supports enrichers to augment index records with additional information, such as metadata derived from the source data or advanced operations like language detection. You can specify enrichers during indexing for enhanced functionality:

```bash
dextro create-index dataset/ --enrichers=detect_language
```

### Data Filtering

Dextro allows for advanced data filtering directly on the index, facilitating efficient data selection without explicit loading:

```python
import polars as pl
from dextro.dataset import IndexedDataset

# Example filter: Select texts within a specific character length range
# This assumes that the `TextLength` enricher has been used during indexing
dataset = IndexedDataset(
    data_root='dataset/',
    index_filter=(256 <= pl.col('meta_text_length')) & (pl.col('meta_text_length') <= 1024)
)
```

### Non-NLP Datasets

Dextro can in principle work with any data modality as it this doesn't make assumptions about the data representation. 

### Other Data Formats

With the default settings, Dextro assumes that the dataset is formatted in JSON Lines format. Other formats are supporte via the `load_fn` option of the `FileIndexer` class. However, records currently have to be separated by lines.

## Examples

COMING SOON

## Development

### Install Dev Dependencies

```bash
poetry install --all-extras --with=dev
```

### Run Tests

```bash
pytest tests
```

### Autoformat


```bash
ruff format .
```

## Why "Dextro"?

The name "Dextro" is inspired by dextrose, a historic term for glucose and associated with fast energy delivery. This name reflects the toolkit's aim to provide fast, efficient processing and low overhead for dataset handling, mirroring the quick energy boost dextrose is known for.

Dextro is designed to be the optimal solution for managing and accessing large language datasets, enabling rapid and flexible data handling to support the advancement of AI and machine learning research.
