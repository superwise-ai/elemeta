from typing import Optional

import nltk  # type: ignore

from elemeta.nlp.extractors.low_level.hinted_profanity_token_count import HintedProfanityTokensCount


class HintedProfanitySentenceCount(HintedProfanityTokensCount):
    """
    Counts the number of sentences with profanity words in them uses better_profanity library. https://github.com/snguyenthanh/better_profanity

    Parameters
    ----------
    name : str, optional
        Name of the metadata. If not given, the name will be extracted from the class name.

    Examples
    --------
    >>> profanity_counter = HintedProfanitySentenceCount()
    >>> text = "Fuck this sh!t. I want to fucking leave the country, but I am fine"
    >>> profanity_count = profanity_counter(text)
    >>> print(profanity_count) #Output: 1
    """

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, tokenizer=nltk.sent_tokenize)
