from dextro.enrichers import TextLength, LanguageDetectionEnricher
from lingua import Language


def test_test_length_enricher(file_item):
    enricher = TextLength()
    file_item = enricher.enrich_item(file_item)
    item_meta = file_item.meta

    assert "text_length" in item_meta.additional_info
    assert item_meta.additional_info["text_length"] == len(file_item.data["text"])


def test_language_detection_enricher(file_item_batch):
    enricher = LanguageDetectionEnricher(languages=[Language.ENGLISH, Language.LATIN])
    file_item_batch = enricher.enrich_batch(file_item_batch)

    for file_item in file_item_batch:
        item_meta = file_item.meta
        assert "language" in item_meta.additional_info
        assert item_meta.additional_info["language"] == "LA"
