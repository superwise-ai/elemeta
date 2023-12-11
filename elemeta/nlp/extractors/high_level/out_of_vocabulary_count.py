from typing import Optional, Set

from nltk import RegexpTokenizer  # type: ignore
from nltk.corpus import words  # type: ignore

from elemeta.nlp.extractors.low_level.tokens_count import TokensCount


class OutOfVocabularyCount(TokensCount):
    """
    For a given vocabulary (the default is English vocabulary taken from nltk.corpus)
    return the number of words outside of the vocabulary

    Example
    -------
    >>> from elemeta.nlp.extractors.high_level.out_of_vocabulary_count import OutOfVocabularyCount
    >>> text = "Rick said Wubba Lubba dub-dub"
    >>> oov_counter = OutOfVocabularyCount()
    >>> print(oov_counter(text)) #Output: 3
    """

    def __init__(
        self,
        vocabulary: Optional[Set[str]] = None,
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metafeature of not given will extract the name from the class name
        vocabulary: Optional[Set[str]]
            set of words defined as known words
        """
        super().__init__(
            name=name,
            tokenizer=RegexpTokenizer(r"""\w(?<!\d)[\w'-]*""").tokenize,
            exclude_tokens_list=vocabulary or set(words.words()),
        )

    def extract(self, text: str) -> int:
        return super().extract(text=text.lower())
