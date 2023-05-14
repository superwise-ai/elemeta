from typing import Callable, List, Optional

from nltk import word_tokenize  # type: ignore

from elemeta.nlp import english_stopwords
from elemeta.nlp.extractors.low_level.tokens_count import TokensCount


class StopWordsCount(TokensCount):
    """
    Counts the number of stop words
    """

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]] = word_tokenize,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, tokenizer=tokenizer, include_tokens_list=english_stopwords)
