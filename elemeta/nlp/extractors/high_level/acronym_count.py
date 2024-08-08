from typing import Optional

from elemeta.nlp.extractors.high_level.regex_match_count import RegexMatchCount


class AcronymCount(RegexMatchCount):
    """
    Counts the number of acronyms in the text.

    Parameters
    ----------
    name : Optional[str], optional
        Name to use for the metadata. If not given, the name will be extracted from the class name.

    Examples
    --------
    >>> from elemeta.nlp.extractors.high_level.acronym_count import AcronymCount
    >>> text = "W.T.F that was LOL"
    >>> counter = AcronymCount()
    >>> result = counter(text)
    >>> print(result) # Output: 2
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the AcronymCount object.

        Parameters
        ----------
        name : Optional[str], optional
            Name to use for the metadata. If not given, the name will be extracted from the class name.
        """
        super().__init__(regex=r"\b(?:[A-Z]\.?)*[A-Z]\b", name=name)  # noqa: W605
