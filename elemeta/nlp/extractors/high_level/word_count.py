from typing import Callable, List, Optional, Set

from nltk import word_tokenize  # type: ignore

from elemeta.nlp import extended_punctuations
from elemeta.nlp.extractors.low_level.tokens_count import TokensCount


class WordCount(TokensCount):
    """
    Gives the number of words in the text
    """

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]] = word_tokenize,
        exclude_tokens_list: Set[str] = extended_punctuations,
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        tokenizer: Callable[[str],List[str]]
            a function that splits a text into components. Usually into words

        exclude_tokens_list: Set[str]
            set of words to exclude
        """
        super().__init__(name=name, tokenizer=tokenizer, exclude_tokens_list=exclude_tokens_list)
