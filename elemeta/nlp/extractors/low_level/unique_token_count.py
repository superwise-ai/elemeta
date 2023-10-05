from typing import Callable, List, Optional, Set
from collections import Counter

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import (
    AbstractTextMetafeatureExtractor,
)


class UniqueTokenCount(AbstractTextMetafeatureExtractor):
    """
    Implementation of AbstractTextMetafeatureExtractor class
    that return the number of unique tokens in the text
    """

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        exclude_tokens_list: Optional[Set[str]] = None,
        include_tokens_list: Optional[Set[str]] = None,
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        tokenizer: Callable[[str],List[str]]
            a function that splits a text into components
        exclude_tokens_list: Optional[Set[str]]
            an optional parameter that define if there is tokens that should be excluded
             from the counting
        include_tokens_list: Optional[Set[str]]
            an optional parameter that define a closed set of tokens that we want to count

        """
        super().__init__(name)
        self.tokenizer = tokenizer
        self.exclude_tokens_list = exclude_tokens_list
        self.include_tokens_list = include_tokens_list

    def extract(self, text: str) -> int:
        """counts the number tokens in the text

        Parameters
        ----------
        text: str
            the text to check appearance on

        Returns
        -------
        int
            the number of appearance of a must-appear word list
        """

        tokens: List[str] = self.tokenizer(text)

        counts: Counter[str] = Counter(tokens)

        if self.exclude_tokens_list:
            for token in self.exclude_tokens_list:
                counts[token] = 0
        elif self.include_tokens_list:
            for token in counts.copy():
                if token not in self.include_tokens_list:
                    counts.pop(token)

        return sum(1 for count in counts.values() if count == 1)
