from typing import Optional, Set

from nltk import word_tokenize  # type: ignore

from elemeta.nlp import english_punctuations
from elemeta.nlp.extractors.low_level.unique_token_count import UniqueTokenCount


class UniqueWordCount(UniqueTokenCount):
    """Currently returns the number of words in the text that appear exactly once,
    will change to count the unique words in the text

    Example
    -------
    >>> from elemeta.nlp.extractors.high_level.unique_word_count import UniqueWordCount
    >>> text = "Once I was afraid, I was petrified"
    >>> unique_word_count = UniqueWordCount()
    >>> result = unique_word_count(text)
    >>> print(result)  # Output: 3
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
            name to of the metafeature of not given will extract the name from the class name
        exceptions: Set[str]
            words to exclude
        """
        super().__init__(name=name, tokenizer=word_tokenize, exclude_tokens_list=exceptions)
