from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import (
    AbstractTextMetafeatureExtractor,
)


class TextLength(AbstractTextMetafeatureExtractor):
    """
    Gives the number of characters in the text (including whitespace).
    """

    def extract(self, text: str) -> int:
        """
        text length counter
        returns the length of the text

        Parameters
        ----------
        text: str
            the text to check length on

        Returns
        -------
        int
            the length of the text

        """
        return len(text)
