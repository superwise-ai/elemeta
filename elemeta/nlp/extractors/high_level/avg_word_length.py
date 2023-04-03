from typing import Callable, List, Optional, Set

from nltk import word_tokenize  # type: ignore

from elemeta.nlp import extended_punctuations
from elemeta.nlp.extractors.low_level.avg_token_length import AvgTokenLength


class AvgWordLength(AvgTokenLength):
    """
    Gives the average length of the words in the text
    """

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]] = word_tokenize,
        exclude_list: Set[str] = extended_punctuations,
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        tokenizer: Callable[[str],List[str]]
            a function that splits a text into components
        exclude_list: Set[str]
           set of words to exclude when computing the metric
        """
        super().__init__(name=name, tokenizer=tokenizer, tokens_to_exclude=exclude_list)
