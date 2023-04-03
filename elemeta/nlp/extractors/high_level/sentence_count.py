from typing import Callable, List, Optional

import nltk  # type: ignore

from elemeta.nlp.extractors.low_level.tokens_count import TokensCount


class SentenceCount(TokensCount):
    """
    Counts the number of sentences in the text
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
            a function that splits a text into components
        """
        super().__init__(name=name, tokenizer=tokenizer)
        self.tokenizer = tokenizer
