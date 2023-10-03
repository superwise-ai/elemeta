from typing import Callable, List, Optional, Set

from nltk import word_tokenize  # type: ignore

from elemeta.nlp import extended_punctuations
from elemeta.nlp.extractors import length_check_basic
from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import (
    AbstractTextMetafeatureExtractor,
)


class PunctuationCount(AbstractTextMetafeatureExtractor):
    """
    Counts the number of punctuation marks in the text
    """

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]] = word_tokenize,
        punctuations: Set[str] = extended_punctuations,
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        tokenizer: Callable[[str],List[str]]
            a function that splits a text into components. Usually into words
        punctuations: Set()
            set of punctuations

        """

        super().__init__(name)
        self.tokenizer = tokenizer
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
        return length_check_basic(self.tokenizer, lambda token: token in self.punctuations)(text)
