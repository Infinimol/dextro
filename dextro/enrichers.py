from abc import abstractmethod, ABC
from typing import Iterator
from dextro.types import FileItem


class BaseEnricher(ABC):
    @abstractmethod
    def enrich_item(self, item: FileItem) -> FileItem | None:
        pass


class BaseBatchedEnricher(ABC):
    @abstractmethod
    def enrich_batch(
        self, items: list[FileItem]
    ) -> Iterator[FileItem | None]:
        pass


Enricher = BaseEnricher | BaseBatchedEnricher


class TextLength(BaseEnricher):
    """
    Enricher that adds the length of the text as 'meta_text_length' to the item.

    Args:
        text_key: The key of the text field in the serialized item. Defaults to 'text'.
    """

    def __init__(self, text_key: str = "text"):
        self.text_key = text_key

    def enrich_item(self, item: FileItem) -> FileItem:
        item.add_info('text_length', len(item.data[self.text_key]))
        return item


class LanguageDetectionEnricher(BaseBatchedEnricher):
    """
    Enricher that adds the detected language as 'meta_language' to the item.

    It uses [lingua-py](https://github.com/pemistahl/lingua-py) as the language detection library.

    Args:
        languages: A list of language codes to detect. Defaults to None (all languages).
        detect_multiple: Whether to detect multiple languages.
            If multiple languages are detected, they will be comma-separated. Defaults to False.
        low_accuracy_mode: Whether to use low accuracy mode.
            Results in greater throughput but lower accuracy. Defaults to False.
    """

    def __init__(
        self,
        languages: list[str] | str | None = None,
        detect_multiple: bool = False,
        low_accuracy_mode: bool = False,
        text_key: str = "text",
    ):
        try:
            from lingua import LanguageDetectorBuilder
        except ImportError as e:
            raise ImportError(
                "Please install the lingua-language-detector to support language detection"
            ) from e

        if isinstance(languages, str):
            languages = [languages]

        self.languages = languages
        self.detect_multiple = detect_multiple
        self.low_accuracy_mode = low_accuracy_mode
        self.text_key = text_key

        builder = LanguageDetectorBuilder

        if not languages:
            builder = builder.from_all_languages()
        else:
            builder = builder.from_languages(*languages)

        if low_accuracy_mode:
            builder = builder.with_low_accuracy_mode()

        self.detector = builder.build()

    def enrich_batch(
        self, items: list[FileItem]
    ) -> Iterator[FileItem | None]:
        texts = [item.data[self.text_key] for item in items]

        if self.detect_multiple:
            results = self.detector.detect_multiple_languages_in_parallel_of(texts)

            for result, item in zip(results, items):
                langs = [res.language.iso_code_639_1.name for res in result]

                if langs:
                    langs = ",".join(langs)
                else:
                    langs = None

                item.add_info("language", sorted(langs))
        else:
            langs = self.detector.detect_languages_in_parallel_of(texts)

            for lang, item in zip(langs, items):
                item.add_info("language", lang.iso_code_639_1.name)

        return items


enricher_registry = {
    "text_length": lambda: TextLength(),
    "detect_language": lambda: LanguageDetectionEnricher(),
    "detect_language_low_accuracy": lambda: LanguageDetectionEnricher(
        low_accuracy_mode=True
    ),
    "detect_languages": lambda: LanguageDetectionEnricher(detect_multiple=True),
    "detect_languages_low_accuracy": lambda: LanguageDetectionEnricher(
        detect_multiple=True, low_accuracy_mode=True
    ),
}
