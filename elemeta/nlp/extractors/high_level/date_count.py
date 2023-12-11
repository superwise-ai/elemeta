from typing import Optional

import datefinder  # type: ignore

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import AbstractTextMetafeatureExtractor


class DateCount(AbstractTextMetafeatureExtractor):
    """
    Counts the number of dates in the text.

    Parameters
    ----------
    name : Optional[str], optional
        Name of the metafeature. If not given, the name will be extracted from the class name.

    Attributes
    ----------
    name : str
        Name of the metafeature.

    Methods
    -------
    extract(text)
        Return the number of dates in the text.

    Examples
    --------
    >>> date_counter = DateCount()
    >>> text = "Entries are due by January 4th, 2017 at 8:00pm, created 01/15/2005 by ACME Inc. and associates."
    >>> date_counter(text) #Output: 2
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def extract(self, text: str) -> int:
        """
        Return the number of dates in the text.

        Parameters
        ----------
        text : str
            The string to run on.

        Returns
        -------
        int
            The number of dates in the text.
        """
        return len(list(datefinder.find_dates(text)))
