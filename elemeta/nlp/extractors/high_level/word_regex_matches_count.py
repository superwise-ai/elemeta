from typing import Callable, List, Optional

from nltk import word_tokenize  # type: ignore

from elemeta.nlp.extractors.low_level.regex_token_matches_count import TokenRegexMatchesCount


class WordRegexMatchesCount(TokenRegexMatchesCount):
    """
    For a given regex return the number of words matching the regex
    """

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]] = word_tokenize,
        regex: str = ".*",
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        tokenizer: Callable[[str],List[str]]
            a function that splits a text into components
        regex: str
            regex to try to match
        """

        super().__init__(name=name, tokenizer=tokenizer, regex=regex)
