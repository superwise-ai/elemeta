from typing import Callable, List, Optional

import nltk  # type: ignore

from elemeta.nlp.extractors.low_level.hinted_profanity_token_count import (
    HintedProfanityTokensCount,
)


class HintedProfanitySentenceCount(HintedProfanityTokensCount):
    """
    Counts the number of sentences with profanity words in them
    """

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]] = nltk.sent_tokenize,
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
