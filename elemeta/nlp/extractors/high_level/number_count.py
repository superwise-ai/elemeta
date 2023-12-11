from typing import Optional

from nltk import word_tokenize  # type: ignore

from elemeta.nlp.extractors import length_check_basic
from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import AbstractTextMetafeatureExtractor


class NumberCount(AbstractTextMetafeatureExtractor):
    """
    Counts the number of numbers in the text.

    Parameters
    ----------
    name : str, optional
        Name of the metafeature. If not given, the name will be extracted from the class name.

    Examples
    --------
    >>> from elemeta.nlp.extractors.high_level.number_count import NumberCount
    >>> number_counter = NumberCount()
    >>> text = "There are 3 apples and 5 oranges."
    >>> number_counter(text) #Output: 2
    """

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(name)

    def validator(self, token: str) -> bool:
        """
        Number check validator.
        Checks if the token is a number.

        Parameters
        ----------
        token : str
            The token to check if it is a number.

        Returns
        -------
        bool
            True if the token is a number.
        """
        return token.isnumeric()

    def extract(self, text: str) -> int:
        """
        Return the number of numbers in the text.

        Parameters
        ----------
        text : str
            The string to run on.

        Returns
        -------
        int
            The number of numbers in the text.
        """
        return length_check_basic(word_tokenize, self.validator)(text)
