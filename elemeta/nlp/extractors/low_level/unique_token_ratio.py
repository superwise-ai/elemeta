from typing import Callable, List, Optional, Set

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import (
    AbstractTextMetafeatureExtractor,
)


class UniqueTokensRatio(AbstractTextMetafeatureExtractor):
    """Implementation of AbstractTextMetafeatureExtractor class that return the ratio between the
    number of unique tokens to all tokens"""

    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        exceptions: Set[str],
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        tokenizer: Callable[[str],List[str]]
            a function that splits a text into components
        exceptions: Set[str]
            tokens to exclude
        """
        super().__init__(name)
        self.tokenizer = tokenizer
        self.exceptions = exceptions

    def extract(self, text: str) -> float:
        """Unique words in text function

        returns the ratio between set(tokens)/len(tokens)
        filters on tokens that are defined as relevant


        Parameters
        ----------
        text: str
            the text we want to find unique words ratio on

        Returns
        -------
        sentiment: float
            the ratio between len(set(tokens that appear once ))/len(set(tokens))

        """
        tokens = self.tokenizer(text)
        corpus = list(filter(lambda x: x not in self.exceptions, tokens))
        counts = {token: 0 for token in corpus}
        for token in corpus:
            counts[token] += 1
        unique_tokes = list(filter(lambda k: counts[k] == 1, counts))
        if len(counts.keys()) == 0:
            return 0
        return len(unique_tokes) / len(counts.keys())
