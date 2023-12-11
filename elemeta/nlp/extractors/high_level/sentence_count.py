from typing import Optional

import nltk  # type: ignore

from elemeta.nlp.extractors.low_level.tokens_count import TokensCount


class SentenceCount(TokensCount):
    """
    Counts the number of sentences in the text

    Example
    -------
    >>> from elemeta.nlp.extractors.high_level.sentence_count import SentenceCount
    >>> text = "Hello, my name is Inigo Montoya. You killed my father. Prepare to die."
    >>> sentence_count = SentenceCount()
    >>> result = sentence_count(text)
    >>> print(result)  # Output: 3
    """

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        """
        super().__init__(name=name, tokenizer=nltk.sent_tokenize)
