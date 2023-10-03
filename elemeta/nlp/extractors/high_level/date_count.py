from typing import Optional

import datefinder  # type: ignore

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import (
    AbstractTextMetafeatureExtractor,
)


class DateCount(AbstractTextMetafeatureExtractor):
    """
    Counts the number of dates in the text
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
        """
        return the number of dates in the text

        Parameters
        ----------
        text: str
            the string to run on
        Returns
        -------
        int
            the number of dates in the text
        """
        return len(list(datefinder.find_dates(text)))
