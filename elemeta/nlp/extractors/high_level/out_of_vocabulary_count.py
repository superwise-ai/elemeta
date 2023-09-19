from typing import Callable, List, Optional, Set

from nltk import RegexpTokenizer  # type: ignore
from nltk.corpus import words  # type: ignore

from elemeta.nlp.extractors.low_level.tokens_count import TokensCount


class OutOfVocabularyCount(TokensCount):
    """
    For a given vocabulary (the default is English vocabulary taken from nltk)
    return the number of words outside of the vocabulary
    """

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]] = RegexpTokenizer(r"""\w(?<!\d)[\w'-]*""").tokenize,
        vocabulary: Optional[Set[str]] = None,
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        tokenizer: Callable[[str],List[str]]
            a function that splits a text into components
        vocabulary: Optional[Set[str]]
            set of words defined as known words
        """
        super().__init__(
            name=name,
            tokenizer=tokenizer,
            exclude_tokens_list=vocabulary or set(words.words()),
        )

    def extract(self, text: str) -> int:
        return super().extract(text=text.lower())
