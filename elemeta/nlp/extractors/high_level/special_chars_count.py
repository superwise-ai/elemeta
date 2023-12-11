from typing import Optional, Set

from nltk import word_tokenize  # type: ignore

from elemeta.nlp import special_chars
from elemeta.nlp.extractors.low_level.tokens_count import TokensCount


class SpecialCharsCount(TokensCount):
    """
    Counts the number of special characters in the .

    Example
    -------
    >>> from elemeta.nlp.extractors.high_level.special_chars_count import SpecialCharsCount
    >>> text = "Once I was afraid, I was petrified!"
    >>> special_chars_count = SpecialCharsCount()
    >>> result = special_chars_count(text)
    >>> print(result)  # Output: 1
    """

    def __init__(
        self,
        specials: Set[str] = special_chars,
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        specials: set()
            a set of special characters (,.!@#$...)
        """
        super().__init__(name=name, tokenizer=word_tokenize, include_tokens_list=specials)
