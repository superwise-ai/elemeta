from typing import Optional

from elemeta.nlp.extractors.high_level.regex_match_count import RegexMatchCount


class EmailCount(RegexMatchCount):
    """
    Counts the number of emails in the text.

    Parameters
    ----------
    name : Optional[str], optional
        Name to use for the metafeature. If not given, the name will be extracted from the class name.

    Examples
    --------
    >>> email_counter = EmailCount(name="email_count")
    >>> text = "lior.something@gmail.ac.il is ok but lior@superwise.il is better"
    >>> count = email_counter(text)
    >>> print(count) #Output: 2
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(regex="[\w\-\.\+]+@([\w-]+\.)+[\w-]{2,4}", name=name)  # noqa

    def extract(self, input: str) -> int:
        return super().extract(input)
