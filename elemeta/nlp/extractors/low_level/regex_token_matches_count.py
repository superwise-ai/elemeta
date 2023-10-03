import re
from typing import Callable, List, Optional

from elemeta.nlp.extractors import length_check_basic
from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import (
    AbstractTextMetafeatureExtractor,
)


class TokenRegexMatchesCount(AbstractTextMetafeatureExtractor):
    """Implementation of AbstractTextMetafeatureExtractor class that return number of tokens
    that match the given regex"""

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        regex: str = ".*",
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        tokenizer: Callable[[str],List[str]]
            a function that splits a text into components
        regex: str
            regex to try to match
        """

        super().__init__(name)
        self.tokenizer = tokenizer
        self.regex = regex

    def validator(self, token: str) -> bool:
        """regex check validator
        checks if the token abides by the regex

        Parameters
        ----------
        token: str
            the token check if abides by the regex

        Returns
        -------
        bool
            true if the token abides and false otherwise

        """
        return bool(re.fullmatch(self.regex, token))

    def extract(self, text: str) -> int:
        """
        return the number of matches of the given regex in the text

        Parameters
        ----------
        text: str
            the string to run on
        Returns
        -------
        int
            the number of the given text in the text
        """
        return length_check_basic(self.tokenizer, self.validator)(text)
