from typing import Optional, Set

from nltk import word_tokenize  # type: ignore

from elemeta.nlp import extended_punctuations
from elemeta.nlp.extractors.low_level.avg_token_length import AvgTokenLength


class AvgWordLength(AvgTokenLength):
    """
    Gives the average length of the words in the text.

    Parameters
    ----------
    exclude_list : Set[str], optional
        Set of words to exclude when computing the metric. Default is `extended_punctuations`.
    name : str, optional
        Name of the metafeature. If not given, it will extract the name from the class name.

    Example
    -------
    >>> from elemeta.nlp.extractors.high_level.avg_word_length import AvgWordLength
    >>> text = "Hello, my name is Inigo Montoya. You killed my father. Prepare to die."
    >>> avg_word_length = AvgWordLength()
    >>> result = avg_word_length(text)
    >>> print(result)  # Output: 4.538
    """

    def __init__(
        self,
        exclude_list: Set[str] = extended_punctuations,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, tokenizer=word_tokenize, tokens_to_exclude=exclude_list)
