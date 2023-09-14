from typing import Callable, List, Optional, Set

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import (
    AbstractTextMetafeatureExtractor,
)


class MustAppearTokensPercentage(AbstractTextMetafeatureExtractor):
    """Implementation of AbstractTextMetafeatureExtractor class that return the ration between
    the number of appearances of tokens from

    given tokens list in the text to all the tokens"""

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        must_appear: Set[str],
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        tokenizer: Callable[[str],List[str]]
            a function that splits a text into components
        must_appear: Optional[Set[str]]
            set of tokes that must appear
        """
        super().__init__(name)
        self.tokenizer = tokenizer
        self.must_appear = must_appear

    def extract(self, text: str) -> float:
        """gives the percentage of the tokens in must_appear set that appeared in the text

        Parameters
        ----------
        text: str
            the text to check appearance on

        Returns
        -------
        float
            the ratio between the number of must-appear tokens to all words
        """
        tokens = self.tokenizer(text)
        corpus = [token for token in tokens if token in self.must_appear]
        if len(self.must_appear) == 0:
            return 0
        return len(set(corpus)) / len(self.must_appear)
