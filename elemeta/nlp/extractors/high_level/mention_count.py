from typing import Optional

from elemeta.nlp.extractors.high_level.regex_match_count import RegexMatchCount


class MentionCount(RegexMatchCount):
    """
    Counts the number of mentions (word in the format @someones_name)

    Parameters
    ----------
    name : Optional[str], optional
        Name to use for the metafeature. If not given, the name will be extracted from the class name.

    Examples
    --------
    >>> from elemeta.nlp.extractors.high_level.mention_count import MentionCount
    >>> mention_counter = MentionCount()
    >>> count = mention_counter("Hello @JohnDoe, how are you?")
    >>> print(count) #Output: 1
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the MentionCount extractor.

        Parameters
        ----------
        name : Optional[str], optional
            Name to use for the metadata. If not given, the name will be extracted from the class name.
        """
        super().__init__(regex="(^|[^@\w])@(\w+)", name=name)  # noqa: W605
