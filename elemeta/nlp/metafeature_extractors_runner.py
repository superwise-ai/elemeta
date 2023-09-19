from typing import Any, Dict, List, Optional

from pandas import DataFrame

from elemeta.nlp.extractors.high_level.acronym_count import AcronymCount
from elemeta.nlp.extractors.high_level.avg_word_length import AvgWordLength
from elemeta.nlp.extractors.high_level.capital_letters_ratio import CapitalLettersRatio
from elemeta.nlp.extractors.high_level.date_count import DateCount
from elemeta.nlp.extractors.high_level.detect_langauge_langdetect import DetectLanguage
from elemeta.nlp.extractors.high_level.email_count import EmailCount
from elemeta.nlp.extractors.high_level.emoji_count import EmojiCount
from elemeta.nlp.extractors.high_level.hashtag_count import HashtagCount
from elemeta.nlp.extractors.high_level.hinted_profanity_sentence_count import (
    HintedProfanitySentenceCount,
)
from elemeta.nlp.extractors.high_level.hinted_profanity_words_count import (
    HintedProfanityWordsCount,
)
from elemeta.nlp.extractors.high_level.link_count import LinkCount
from elemeta.nlp.extractors.high_level.mention_count import MentionCount
from elemeta.nlp.extractors.high_level.must_appear_words_percentage import (
    MustAppearWordsPercentage,
)
from elemeta.nlp.extractors.high_level.number_count import NumberCount
from elemeta.nlp.extractors.high_level.out_of_vocabulary_count import OutOfVocabularyCount
from elemeta.nlp.extractors.high_level.punctuation_count import PunctuationCount
from elemeta.nlp.extractors.high_level.regex_match_count import RegexMatchCount
from elemeta.nlp.extractors.high_level.sentence_avg_length import SentenceAvgLength
from elemeta.nlp.extractors.high_level.sentence_count import SentenceCount
from elemeta.nlp.extractors.high_level.sentiment_polarity import SentimentPolarity
from elemeta.nlp.extractors.high_level.sentiment_subjectivity import SentimentSubjectivity
from elemeta.nlp.extractors.high_level.special_chars_count import SpecialCharsCount
from elemeta.nlp.extractors.high_level.stop_words_count import StopWordsCount
from elemeta.nlp.extractors.high_level.syllable_count import SyllableCount
from elemeta.nlp.extractors.high_level.text_complexity import TextComplexity
from elemeta.nlp.extractors.high_level.text_length import TextLength
from elemeta.nlp.extractors.high_level.unique_word_count import UniqueWordCount
from elemeta.nlp.extractors.high_level.unique_word_ratio import UniqueWordRatio
from elemeta.nlp.extractors.high_level.word_count import WordCount
from elemeta.nlp.extractors.high_level.word_regex_matches_count import WordRegexMatchesCount
from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import (
    AbstractTextMetafeatureExtractor,
)

intensive_metrics = [
    SentimentPolarity(),
    SentimentSubjectivity(),
    HintedProfanityWordsCount(),
    HintedProfanitySentenceCount(),
]
non_intensive_metrics = [
    EmojiCount(),
    TextComplexity(),
    UniqueWordRatio(),
    UniqueWordCount(),
    WordRegexMatchesCount(),
    NumberCount(),
    OutOfVocabularyCount(),
    MustAppearWordsPercentage(must_appear=set()),
    SentenceCount(),
    SentenceAvgLength(),
    WordCount(),
    AvgWordLength(),
    TextLength(),
    StopWordsCount(),
    PunctuationCount(),
    SpecialCharsCount(),
    CapitalLettersRatio(),
    RegexMatchCount(),
    EmailCount(),
    LinkCount(),
    HashtagCount(),
    MentionCount(),
    SyllableCount(),
    AcronymCount(),
    DateCount(),
    DetectLanguage(),
]


class MetafeatureExtractorsRunner:
    """
    This class used to run multiple MetadataExtractors on a text

    Attributes
    ----------
    metafeature_extractors : Optional[List[AbstractTextMetafeatureExtractor]]
        a list of `MetadataExtractor`s to run,
        if not supplied will run with all metadata extractors.

    Methods
    -------
    run(text)
        runs all the metadata extractors on the input text
    run_on_dataframe(df,text_column)
        runs all the metadata extractors on the given text_column in the given dataframe
         and return new dataframe with metadata values as columns
    """

    def __init__(
        self,
        metafeature_extractors: Optional[List[AbstractTextMetafeatureExtractor]] = None,
        compute_intensive: bool = False,
    ):
        """Representation of a df, text column, and list of `AbstractMetadataExtractor` to run on
        This is a wrapper for a pandas df,
        the column holding the text,
        and the `AbstractMetadataExtractor`s to run on.

        Parameters
        ----------
        metafeature_extractors : Optional[List[AbstractTextMetafeatureExtractor]]
            a list of `AbstractMetadataExtractor`s to run over. Runs on all of them independently.
            if not supplied will initialize a list of all metrics with the default configuration

        """
        if metafeature_extractors is not None:
            self.metafeature_extractors = metafeature_extractors
        elif compute_intensive:
            self.metafeature_extractors = non_intensive_metrics + intensive_metrics

        else:
            self.metafeature_extractors = non_intensive_metrics.copy()

    def run(self, text: str) -> Dict[str, Any]:
        """run metrics on list of text

        Parameters
        ----------
        text: str
            the text to run all metrics on

        Returns
        -------
        metafeature_value_dict: Dict[str, Any]
            returns a dictionary of extractor name and the metafeature value

        """
        return {metric.name: metric.extract(text) for metric in self.metafeature_extractors}

    def run_on_dataframe(self, dataframe: DataFrame, text_column: str) -> DataFrame:
        """return new dataframe with all metafeature extractors values
        Parameters
        ----------
        dataframe: DataFrame
            dataframe with the text column
        text_column: str
            the name of the text column in the given dataframe

        Returns
        -------
        dataframe: DataFrame
            dataframe with the values of the metafeature extractors as new columns
        """
        dataframe_to_return = dataframe.copy()
        if text_column not in dataframe_to_return.columns:
            raise AssertionError(
                f"The given text_column:'{text_column}' doesn't exist in the given dataframe"
            )

        names = set()
        for metric in self.metafeature_extractors:
            assert metric.name not in names, f"more than one metric have the name {metric.name}"
            names.add(metric.name)

        data_frame_text = dataframe_to_return[text_column]
        for metric in self.metafeature_extractors:
            dataframe_to_return.loc[:, metric.name] = data_frame_text.map(metric.extract)

        return dataframe_to_return

    def add_metafeature_extractor(
        self, metafeature_extractor: AbstractTextMetafeatureExtractor
    ) -> None:
        self.metafeature_extractors.append(metafeature_extractor)
