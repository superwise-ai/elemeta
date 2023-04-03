from typing import Callable, List, Optional, Set

from nltk import word_tokenize  # type: ignore

from elemeta.nlp.extractors.low_level.must_appear_tokens_parentage import (
    MustAppearTokensPercentage,
)


class MustAppearWordsPercentage(MustAppearTokensPercentage):
    """
    For a given set of words, return the percentage of words that appeared in the text
    """

    def __init__(
        self,
        must_appear: Set[str],
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
        must_appear: Optional[Set[str]]
            set of words that must appear
        """
        super().__init__(name=name, tokenizer=tokenizer, must_appear=must_appear)
