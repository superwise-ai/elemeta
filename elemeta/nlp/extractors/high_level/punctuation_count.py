from typing import Optional, Set

from nltk import word_tokenize  # type: ignore

from elemeta.nlp import extended_punctuations
from elemeta.nlp.extractors import length_check_basic
from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import AbstractTextMetafeatureExtractor


class PunctuationCount(AbstractTextMetafeatureExtractor):
    """
    Counts the number of punctuation marks in the text

    Example
    -------
    >>> from elemeta.nlp.extractors.high_level.punctuation_count import PunctuationCount
    >>> text = "Once I was afraid, I was petrified!"
    >>> punctuation_count = PunctuationCount()
    >>> result = punctuation_count(text)
    >>> print(result)  # Output: 2
    """

    def __init__(
        self,
        punctuations: Set[str] = extended_punctuations,
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metafeature of not given will extract the name from the class name
        punctuations: Set()
            set of punctuations

        """

        super().__init__(name)
        self.punctuations = punctuations

    def extract(self, text: str) -> int:
        """
        return the number of punctuations in the text

        Parameters
        ----------
        text: str
            the string to run on
        Returns
        -------
        int
            the number of punctuations in the text
        """
        return length_check_basic(word_tokenize, lambda token: token in self.punctuations)(text)
