from typing import Optional

import emoji

from elemeta.nlp.extractors.low_level.abstract_metadata_extractor import (
    AbstractMetadataExtractor,
)


class EmojiCount(AbstractMetadataExtractor):
    """
    Counts the number of emojis in the text
    """

    def __init__(self, name: Optional[str] = None):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        """
        super().__init__(name)

    def extract(self, text: str) -> int:
        """emoji counter function

        Parameters
        ----------
        text: str
            the text to count emoji on

        Returns
        -------
        int
            the number of emojis in the text

        """
        return emoji.emoji_count(text)
