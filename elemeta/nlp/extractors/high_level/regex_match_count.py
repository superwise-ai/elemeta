import re
from typing import Optional

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import (
    AbstractTextMetafeatureExtractor,
)


class RegexMatchCount(AbstractTextMetafeatureExtractor):
    """
    For a given regex return the number of matches it has in the text
    """

    def __init__(self, regex: str = ".+", name: Optional[str] = None):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        """
        super().__init__(name)
        self.regex = regex

    def extract(self, text: str) -> int:
        """regex count function

        Parameters
        ----------
        input:str
            a text to run the regex on

        Returns
        -------
        int
           how many times the regex is found in the string

        """
        return len(re.findall(self.regex, text))
