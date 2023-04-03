from elemeta.nlp.extractors.low_level.abstract_metadata_extractor import (
    AbstractMetadataExtractor,
)


class TextLength(AbstractMetadataExtractor):
    """
    Gives the length of the text
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
