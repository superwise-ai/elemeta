from typing import Optional

from elemeta.nlp.extractors.low_level.hinted_profanity_token_count import HintedProfanityTokensCount


class HintedProfanityWordsCount(HintedProfanityTokensCount):
    """
    Counts the number of profanity words
    (uses better_profanity better_profanity library).
    https://github.com/snguyenthanh/better_profanity

    Parameters
    ----------
    name : str, optional
        The name of the metafeature. If not given, it will be extracted from the class name.

    Examples
    --------
    >>> from elemeta.nlp.extractors.high_level.hinted_profanity_words_count import HintedProfanityWordsCount
    >>> profanity_word_counter = HintedProfanityWordsCount()
    >>> text = "Fuck this sh!t. I want to fucking leave the country"
    >>> count = profanity_word_counter(text)
    >>> print(count) #Output: 3
    """

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, tokenizer=lambda text: text.split(" "))
