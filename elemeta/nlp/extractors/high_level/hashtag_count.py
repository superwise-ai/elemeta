from typing import Optional

from elemeta.nlp.extractors.high_level.regex_match_count import RegexMatchCount


class HashtagCount(RegexMatchCount):
    """
    Counts the number of hashtags in the text
    """

    def __init__(self, name: Optional[str] = None):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        """
        super().__init__(regex="(\s|^)#(\\w+)", name=name)  # noqa: W605
