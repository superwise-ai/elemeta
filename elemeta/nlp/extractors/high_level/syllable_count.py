from typing import Optional

import textstat  # type: ignore

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import AbstractTextMetafeatureExtractor


class SyllableCount(AbstractTextMetafeatureExtractor):
    """
    Counts the total number of syllables in the text.

    Example
    -------
    >>> from elemeta.nlp.extractors.high_level.syllable_count import SyllableCount
    >>> syllable_count = SyllableCount()
    >>> print(syllable_count("hyperemotionality"))  # Output: 8
    """

    def __init__(self, name: Optional[str] = None):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        """
        super().__init__(name)

    def extract(self, text: str) -> int:
        return textstat.textstat.syllable_count(text)
