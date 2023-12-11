from typing import Optional

import nltk  # type: ignore

from elemeta.nlp.extractors.low_level.avg_token_length import AvgTokenLength


class SentenceAvgLength(AvgTokenLength):
    """
    Gives the average length of sentences in the text

    Example
    -------
    >>> from elemeta.nlp.extractors.high_level.sentence_avg_length import SentenceAvgLength
    >>> texy = "Hello, my name is Inigo Montoya. You killed my father. Prepare to die."
    >>> sentence_avg_length = SentenceAvgLength()
    >>> result = sentence_avg_length(text)
    >>> print(result)  # Output: 22.66668
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
