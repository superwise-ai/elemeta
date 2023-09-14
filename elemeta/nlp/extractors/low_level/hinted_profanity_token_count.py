from typing import Callable, List, Optional

from better_profanity import profanity  # type: ignore

from elemeta.nlp.extractors import length_check_basic
from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import (
    AbstractTextMetafeatureExtractor,
)


class HintedProfanityTokensCount(AbstractTextMetafeatureExtractor):
    """
    Implementation of AbstractTextMetafeatureExtractor class that count the number profanity words
    """

    def __init__(self, tokenizer: Callable[[str], List[str]], name: Optional[str] = None):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        tokenizer: Callable[[str],List[str]]
        a function that splits a text into components.
        """
        super().__init__(name)
        self.tokenizer = tokenizer

    def extract(self, text: str) -> int:
        """
        return the number of profanity words in the text

        Parameters
        ----------
        text: str
            the string to run on
        Returns
        -------
        int
            the number of profanity words in the text
        """
        return length_check_basic(
            self.tokenizer, lambda token: profanity.contains_profanity(token)
        )(text)
