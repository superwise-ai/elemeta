from typing import Optional

from nltk import word_tokenize  # type: ignore

from elemeta.nlp import english_stopwords
from elemeta.nlp.extractors.low_level.tokens_count import TokensCount


class StopWordsCount(TokensCount):
    """
    Counts the number of stop words.

    Example
    -------
    >>> from elemeta.nlp.extractors.high_level.stop_words_count import StopWordsCount
    >>> text = "Once I was afraid, I was petrified"
    >>> stop_words_count = StopWordsCount()
    >>> result = stop_words_count(text)
    >>> print(result)  # Output: 4
    """

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, tokenizer=word_tokenize, include_tokens_list=english_stopwords)
