from typing import Callable, List, Optional

from elemeta.nlp.extractors.low_level.hinted_profanity_token_count import (
    HintedProfanityTokensCount,
)


class HintedProfanityWordsCount(HintedProfanityTokensCount):
    """
    Counts the number of profanity words
    """

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]] = lambda text: text.split(" "),
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        tokenizer: Callable[[str],List[str]]
        a function that splits a text into components.
        """
        super().__init__(name=name, tokenizer=tokenizer)
