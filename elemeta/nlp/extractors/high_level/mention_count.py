from typing import Optional

from elemeta.nlp.extractors.high_level.regex_match_count import RegexMatchCount


class MentionCount(RegexMatchCount):
    """
    Counts the number of mentions (word in the format @someones_name)
    """

    def __init__(self, name: Optional[str] = None):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        """
        super().__init__(regex="(^|[^@\w])@(\w+)", name=name)  # noqa: W605
