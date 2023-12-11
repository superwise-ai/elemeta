from typing import Optional

from elemeta.nlp.extractors.high_level.regex_match_count import RegexMatchCount


class HashtagCount(RegexMatchCount):
    """
    Counts the number of hashtags in the text.

    Parameters
    ----------
    name : Optional[str], optional
        Name to use for the metafeature. If not given, the name will be extracted from the class name.

    Examples
    --------
    >>> text = "I love #programming and #coding!"
    >>> hastag_counter = HashtagCount()
    >>> count = hastag_counter(text)
    >>> print(count) #Output: 2
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(regex="(\s|^)#(\\w+)", name=name)  # noqa: W605
