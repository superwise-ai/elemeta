from typing import Optional, Set

from nltk import word_tokenize  # type: ignore

from elemeta.nlp import extended_punctuations
from elemeta.nlp.extractors.low_level.tokens_count import TokensCount


class WordCount(TokensCount):
    """
    Gives the number of words in the text.

    Example
    -------
    >>> from elemeta.nlp.extractors.high_level.word_count import WordCount
    >>> text = "I love to move it move it"
    >>> word_count = WordCount()
    >>> result = word_count(text)
    >>> print(result)  # Output: 7
    """

    def __init__(
        self,
        exclude_tokens_list: Set[str] = extended_punctuations,
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        exclude_tokens_list: Set[str]
            set of words to exclude
        """
        super().__init__(name=name, tokenizer=word_tokenize, exclude_tokens_list=exclude_tokens_list)
