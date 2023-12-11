from typing import Optional

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import AbstractTextMetafeatureExtractor


class CapitalLettersRatio(AbstractTextMetafeatureExtractor):
    """
    Counts the ratio of capital letters to all letters

    Parameters
    ----------
    name : str, optional
        Name of the metafeature. If not given, it will be extracted from the class name.

    Attributes
    ----------
    name : str
        Name of the metafeature.

    Methods
    -------
    extract(text)
        Calculates the ratio of capital letters to lower letters in the given text.

    Examples
    --------
    >>> from elemeta.nlp.extractors.high_level.capital_letters_ratio import CapitalLettersRatio
    >>> extractor = CapitalLettersRatio()
    >>> text = "HalF Ok"
    >>> ratio = extractor.extract(text)
    >>> print(ratio) #Output: 0.5
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def extract(self, text: str) -> float:
        """
        Calculates the ratio of capital letters to all letters in the given text.

        Parameters
        ----------
        text : str
            The text to check the ratio on.

        Returns
        -------
        float
            The ratio of capital letters to lower letters.
        """
        alph = list(filter(str.isalpha, text))
        if len(alph) == 0:
            return 0
        return sum(map(str.isupper, alph)) / len(alph)
