from typing import Optional

from elemeta.nlp.extractors.high_level.regex_match_count import RegexMatchCount


class EmailCount(RegexMatchCount):
    """
    Counts the number of emails in the text
    """

    def __init__(self, name: Optional[str] = None):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        """
        super().__init__(regex="[\w\-\.\+]+@([\w-]+\.)+[\w-]{2,4}", name=name)  # noqa

    def extract(self, input: str) -> int:
        return super().extract(input)
