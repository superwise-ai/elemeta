from typing import Optional

from elemeta.nlp.extractors.high_level.regex_match_count import RegexMatchCount


class LinkCount(RegexMatchCount):
    """
    Counts the number of links in the text
    """

    def __init__(self, name: Optional[str] = None):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        """
        super().__init__(
            regex="^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$",  # noqa
            name=name,
        )
