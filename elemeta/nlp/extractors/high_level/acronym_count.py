from typing import Optional

from elemeta.nlp.extractors.high_level.regex_match_count import RegexMatchCount


class AcronymCount(RegexMatchCount):
    """
    Counts the number of acronyms in the text
    """

    def __init__(self, name: Optional[str] = None):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        """
        super().__init__(regex="(([A-Z](.|-)?)+)(\s|$)", name=name)  # noqa: W605
