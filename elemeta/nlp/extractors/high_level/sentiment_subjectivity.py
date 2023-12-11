from typing import Optional

from textblob import TextBlob  # type: ignore

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import AbstractTextMetafeatureExtractor


class SentimentSubjectivity(AbstractTextMetafeatureExtractor):
    """
    Returns the Sentiment Subjectivity (read more about the
    difference between sentiment polarity and
    sentiment subjectivity
    here:https://www.tasq.ai/tasq-question/what-are-polarity-and-subjectivity-in-sentiment-analysis/)
    value as a range between 0 to 1, where 0.0 is utterly objective, and 1.0 is utterly subjective.

    Example
    --------
    >>> from elemeta.nlp.extractors.high_level.sentiment_subjectivity import SentimentSubjectivity
    >>> sentiment_subjectivity = SentimentSubjectivity()
    >>> print(sentiment_subjectivity("I hate cakes!")) #Output: 0.9
    >>> print(sentiment_subjectivity("They all failed the test")) #Output: 0.3
    """

    def __init__(self, name: Optional[str] = None):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        """
        super().__init__(name)

    def extract(self, text: str) -> float:
        """sentiment subjectivity prediction function

        Parameters
        ----------
        text: str
            the text we want sentiment subjectivity to run on

        Returns
        -------
        sentiment: float
            return subjectivity score as a float within the range [0.0, 1.0]
        where 0.0 is very objective and 1.0 is very subjective.
        """
        return TextBlob(text).sentiment.subjectivity
