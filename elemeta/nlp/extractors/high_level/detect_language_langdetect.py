from typing import Optional

from langdetect import DetectorFactory, LangDetectException, detect  # type: ignore

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import AbstractTextMetafeatureExtractor


class DetectLanguage(AbstractTextMetafeatureExtractor):
    """
    Returns the language of the text.

    Parameters
    ----------
    name : str, optional
        Name of the metafeature. If not given, the name will be extracted from the class name.

    Methods
    -------
    extract(text)
        Detects the language of the given text.

    Examples
    --------
    >>> detect_language = DetectLanguage()
    >>> text = "I love cakes. Its the best. Almost like the rest"
    >>> language = detect_language(text)
    >>> print(language) #Output: 'en'
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the DetectLanguage extractor.

        Parameters
        ----------
        name : str, optional
            Name of the metadata. If not given, the name will be extracted from the class name.
        """
        super().__init__(name)
        DetectorFactory.seed = 42

    def extract(self, text: str) -> str:
        """
        Detects the language of the given text.

        Parameters
        ----------
        text : str
            The text to detect the language on.

        Returns
        -------
        str
            The most likely language of the text.
        """
        try:
            return detect(text)
        except LangDetectException:
            return "unknown"
