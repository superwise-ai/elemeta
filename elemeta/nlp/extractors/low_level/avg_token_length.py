from typing import Callable, List, Optional, Set

from elemeta.nlp.extractors import avg_check_basic
from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import AbstractTextMetafeatureExtractor


class AvgTokenLength(AbstractTextMetafeatureExtractor):
    """
    Implementation of AbstractTextMetafeatureExtractor
    class that return the average token length

    Example
    -------
    >>> from elemeta.nlp.extractors.low_level.avg_token_length import AvgTokenLength
    >>> from nltk import word_tokenize
    >>> text = "Once I was afraid, I was petrified"
    >>> avg_token_length = AvgTokenLength(word_tokenize)
    >>> result = avg_token_length(text)
    >>> print(result)  # Output: 3.5
    """

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        tokens_to_exclude: Optional[Set[str]] = None,
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        tokenizer: Callable[[str],List[str]]
            a function that splits a text into components
        tokens_to_exclude: Set[str]
           set of tokens to exclude when computing the metric
        """
        super().__init__(name)
        self.tokenizer = tokenizer
        self.tokens_to_exclude = tokens_to_exclude

    def extract(self, text: str) -> float:
        """
        return the number of average token length in the text

        Parameters
        ----------
        text: str
            the string to run on
        Returns
        -------
        int
            the number of average tokens length in the text
        """
        if self.tokens_to_exclude is None:
            return avg_check_basic(self.tokenizer, lambda _: True)(text)
        else:
            return avg_check_basic(self.tokenizer, lambda token: token not in self.tokens_to_exclude)(text)  # type: ignore
