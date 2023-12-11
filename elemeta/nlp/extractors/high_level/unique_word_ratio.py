from typing import Optional, Set

from nltk import word_tokenize  # type: ignore

from elemeta.nlp import english_punctuations
from elemeta.nlp.extractors.low_level.unique_token_ratio import UniqueTokensRatio


class UniqueWordRatio(UniqueTokensRatio):
    """
    Gives the ratio between the number of distinct words (total number of different
    values regardless how many times it appears in the dataset) to the number
    of unique words (total number of values that only appear once in the dataset).

    Example
    -------
    >>> from elemeta.nlp.extractors.high_level.unique_word_ratio import UniqueWordRatio
    >>> text = "I love to move it move it"
    >>> unique_word_ratio = UniqueWordRatio()
    >>> result = unique_word_ratio(text)
    >>> print(result)  # Output: 0.6
    """

    def __init__(
        self,
        exceptions: Set[str] = english_punctuations,
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        exceptions: Set[str]
            words to exclude
        """
        super().__init__(name=name, tokenizer=word_tokenize, exceptions=exceptions)
