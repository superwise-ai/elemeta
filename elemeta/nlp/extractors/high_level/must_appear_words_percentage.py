from typing import Optional, Set

from nltk import word_tokenize  # type: ignore

from elemeta.nlp.extractors.low_level.must_appear_tokens_parentage import MustAppearTokensPercentage


class MustAppearWordsPercentage(MustAppearTokensPercentage):
    """
    For a given set of words, return the percentage of words that appeared in the text

    Parameters
    ----------
    must_appear : set of str
        Set of words that must appear in the text.
    name : str, optional
        Name of the metafeature. If not given, the name will be extracted from the class name.

    Examples
    --------
    >>> from elemeta.nlp.extractors.high_level.must_appear_words_percentage import MustAppearWordsPercentage
    >>> text = "I am good now"
    >>> calc_word_precentage = MustAppearWordsPercentage(must_appear={"I", "am"})
    >>> percentage = calc_word_precentage(text)
    >>> print(percentage) #Output: 1
    """

    def __init__(
        self,
        must_appear: Set[str],
        name: Optional[str] = None,
    ):
        super().__init__(name=name, tokenizer=word_tokenize, must_appear=must_appear)
