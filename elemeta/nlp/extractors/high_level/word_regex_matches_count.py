from typing import Optional

from nltk import word_tokenize  # type: ignore

from elemeta.nlp.extractors.low_level.regex_token_matches_count import TokenRegexMatchesCount


class WordRegexMatchesCount(TokenRegexMatchesCount):
    """
    For a given regex return the number of words matching the regex

    Example
    -------
    >>> from elemeta.nlp.extractors.high_level.word_regex_matches_count import WordRegexMatchesCount
    >>> text = "he hee is"
    >>> regex = "h.+"
    >>> word_regex_matches_counter = WordRegexMatchesCount(regex=regex)
    >>> result = word_regex_matches_counter(text)
    >>> print(result)  # Output: 2
    """

    def __init__(
        self,
        regex: str = ".*",
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        regex: str
            regex to try to match
        """

        super().__init__(name=name, tokenizer=word_tokenize, regex=regex)
