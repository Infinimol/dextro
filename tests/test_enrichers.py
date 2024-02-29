from dextro.enrichers import TextLength, LanguageDetectionEnricher
from lingua import Language


def test_test_length_enricher(record):
    enricher = TextLength()
    record = enricher.enrich_item(record)

    assert "meta_text_length" in record
    assert record["meta_text_length"] == len(record["text"])


def test_language_detection_enricher(record_batch):
    enricher = LanguageDetectionEnricher(languages=[Language.ENGLISH, Language.LATIN])
    record_batch = enricher.enrich_batch(record_batch)

    for single_record in record_batch:
        assert "meta_language" in single_record
        assert single_record["meta_language"] == "LA"
