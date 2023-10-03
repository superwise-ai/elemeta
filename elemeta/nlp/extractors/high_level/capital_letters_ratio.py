from typing import Optional

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import (
    AbstractTextMetafeatureExtractor,
)


class CapitalLettersRatio(AbstractTextMetafeatureExtractor):
    """
    Counts the ratio of capital letters to all letters
    """

    def __init__(self, name: Optional[str] = None):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        """
        super().__init__(name)

    def extract(self, text: str) -> float:
        """case ratio calculator
        returns the ratio of capital letters / length

        Parameters
        ----------
        text: str
            the text to check the ratio on

        Returns
        -------
        float
            the ratio of capital letters / lower letters

        """
        alph = list(filter(str.isalpha, text))
        if len(alph) == 0:
            return 0
        return sum(map(str.isupper, alph)) / len(alph)
