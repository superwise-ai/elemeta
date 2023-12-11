from typing import Optional

import emoji

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import AbstractTextMetafeatureExtractor


class EmojiCount(AbstractTextMetafeatureExtractor):
    """
    Counts the number of emojis in the text.

    Parameters
    ----------
    name : str, optional
        Name of the metafeature. If not given, the name will be extracted from the class name.

    Attributes
    ----------
    name : str
        Name of the metafeature.

    Methods
    -------
    extract(text)
        Counts the number of emojis in the given text.

    Examples
    --------
    >>> emoji_counter = EmojiCount()
    >>> text = "🤔 word 🙈 text 😌 ."
    >>> num_emojis = emoji_counter(text)
    >>> print(num_emojis) #Output: 3
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def extract(self, text: str) -> int:
        """
        Counts the number of emojis in the given text.

        Parameters
        ----------
        text : str
            The text to count emojis on.

        Returns
        -------
        int
            The number of emojis in the text.
        """
        return emoji.emoji_count(text)
