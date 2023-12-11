from typing import Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import AbstractTextMetafeatureExtractor


class SentimentPolarity(AbstractTextMetafeatureExtractor):
    """
    Returns the Sentiment Polarity (read more about the difference
    between sentiment polarity and sentiment subjectivity
    here: https://www.tasq.ai/tasq-question/what-are-polarity-and-subjectivity-in-sentiment-analysis/)
    value as a range between -1 to 1, where -1 means
    the text is an utterly negative sentiment
    and 1 is an utterly positive sentiment.


    Example
    --------
    >>> from elemeta.nlp.extractors.high_level.sentiment_polarity import SentimentPolarity
    >>> sentiment_polarity = SentimentPolarity()
    >>> print(sentiment_polarity("I love cake!")) #Output: 0.669
    >>> print(sentiment_polarity("I HATE cake!")) #Output: -0.693
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
        """sentiment analysis prediction function

        Parameters
        ----------
        text: str
            the text we want sentiment analysis to run on

        Returns
        -------
        sentiment: float
            between 0 and 1 representing the sentiment.
            0 negative, 1 positive

        """
        sid_obj = SentimentIntensityAnalyzer()
        sentiment_dict = sid_obj.polarity_scores(text)
        sentiment = sentiment_dict["compound"]
        return sentiment
