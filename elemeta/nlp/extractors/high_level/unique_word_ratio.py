from typing import Callable, List, Optional, Set

from nltk import word_tokenize  # type: ignore

from elemeta.nlp import english_punctuations
from elemeta.nlp.extractors.low_level.unique_token_ratio import UniqueTokensRatio


class UniqueWordRatio(UniqueTokensRatio):
    """
    Gives the ratio between the number of distinct words (total number of different
    values regardless how many times it appears in the dataset) to the number
    of unique words (total number of values that only appear once in the dataset).
    """

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]] = word_tokenize,
        exceptions: Set[str] = english_punctuations,
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        tokenizer: Callable[[str],List[str]]
            a function that splits a text into components
        exceptions: Set[str]
            words to exclude
        """
        super().__init__(name=name, tokenizer=tokenizer, exceptions=exceptions)
