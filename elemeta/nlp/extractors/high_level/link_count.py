from typing import Optional

from elemeta.nlp.extractors.high_level.regex_match_count import RegexMatchCount


class LinkCount(RegexMatchCount):
    """
    Counts the number of links in the text.

    Parameters
    ----------
    name : Optional[str], optional
        Name of the metafeature. If not given, the name will be extracted from the class name.

    Examples
    --------
    >>> from elemeta.nlp.extractors.high_level.link_count import LinkCount
    >>> text = "Check out this link: https://www.example.com"
    >>> link_counter = LinkCount()
    >>> count = link_counter(text)
    >>> print(count) # Output: 1
    """

    def __init__(self, name: Optional[str] = None):
        """
        Parameters
        ----------
        name: Optional[str]
            Name of the metafeature. If not given, the name will be extracted from the class name.
        """
        super().__init__(
            regex="^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$",  # noqa
            name=name,
        )
