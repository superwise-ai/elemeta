from typing import Optional

from langdetect import DetectorFactory, LangDetectException, detect  # type: ignore

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import (
    AbstractTextMetafeatureExtractor,
)


class DetectLanguage(AbstractTextMetafeatureExtractor):
    """
    Returns the language of the text
    """

    def __init__(self, name: Optional[str] = None):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        """
        super().__init__(name)
        DetectorFactory.seed = 42

    def extract(self, text: str) -> str:
        """language detection function

        Parameters
        ----------
        text: str
            the text to detect the language on

        Returns
        -------
        str
           the most likely language of the text
        """
        try:
            return detect(text)
        except LangDetectException:
            return "unknown"
