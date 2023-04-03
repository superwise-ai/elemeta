from typing import Callable, List, Optional, Set

from nltk import word_tokenize  # type: ignore

from elemeta.nlp import special_chars
from elemeta.nlp.extractors.low_level.tokens_count import TokensCount


class SpecialCharsCount(TokensCount):
    """
    Counts the number of special characters in the text
    """

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]] = word_tokenize,
        specials: Set[str] = special_chars,
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        tokenizer: Callable[[str],List[str]]
            a function that splits a text into components. Usually into words
        specials: set()
            a set of special characters (,.!@#$...)
        """
        super().__init__(name=name, tokenizer=tokenizer, include_tokens_list=specials)
