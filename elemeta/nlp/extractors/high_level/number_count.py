from typing import Callable, List, Optional

from nltk import word_tokenize  # type: ignore

from elemeta.nlp.extractors import length_check_basic
from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import (
    AbstractTextMetafeatureExtractor,
)


class NumberCount(AbstractTextMetafeatureExtractor):
    """
    Counts the number of numbers in the text
    """

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]] = word_tokenize,
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        tokenizer: Callable[[str],List[str]]
            a function that splits a text into components
        """

        super().__init__(name)
        self.tokenizer = tokenizer

    def validator(self, token: str) -> bool:
        """number check validator
        checks if the token is a number

        Parameters
        ----------
        token: str
            the token check if is a number

        Returns
        -------
        bool
            true if the token is a number

        """
        return token.isnumeric()

    def extract(self, text: str) -> int:
        """
        return the number of numbers in the text

        Parameters
        ----------
        text: str
            the string to run on
        Returns
        -------
        int
            the number of numbers in the text
        """
        return length_check_basic(self.tokenizer, self.validator)(text)
