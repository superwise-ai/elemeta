import re
from typing import Optional

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import AbstractTextMetafeatureExtractor


class RegexMatchCount(AbstractTextMetafeatureExtractor):
    """
    For a given regex, return the number of matches it has in the text.

    Parameters
    ----------
    regex : str
        The regular expression pattern to match.
    name : Optional[str], optional
        The name of the metafeature. If not given, the name will be extracted from the class name.

    Examples
    --------
    >>> digit_counter = RegexMatchCount(regex=r'\\d', name='Digit Count')
    >>> text = 'There are 3 apples and 52 oranges.'
    >>> digit_counter(text) #Output: 3

    """

    def __init__(self, regex: str = ".+", name: Optional[str] = None):
        """
        Initialize the RegexMatchCount extractor.

        Parameters
        ----------
        regex : str
            The regular expression pattern to match.
        name : Optional[str], optional
            The name of the metafeature. If not given, the name will be extracted from the class name.

        """
        super().__init__(name)
        self.regex = regex

    def extract(self, text: str) -> int:
        """
        Extract the count of matches for the given regex in the text.

        Parameters
        ----------
        text : str
            The text to run the regex on.

        Returns
        -------
        int
            The number of times the regex is found in the string.

        """
        return len(re.findall(self.regex, text))
