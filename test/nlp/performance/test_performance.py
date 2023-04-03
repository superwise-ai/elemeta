import os
from datetime import datetime

import pandas
import pytest

import elemeta.nlp.metadata_extractor_runner as met
from elemeta.nlp.extractors.high_level.acronym_count import AcronymCount
from elemeta.nlp.extractors.high_level.avg_word_length import AvgWordLength
from elemeta.nlp.extractors.high_level.capital_letters_ratio import CapitalLettersRatio
from elemeta.nlp.extractors.high_level.date_count import DateCount
from elemeta.nlp.extractors.high_level.detect_langauge_langdetect import DetectLangauge
from elemeta.nlp.extractors.high_level.email_count import EmailCount
from elemeta.nlp.extractors.high_level.emoji_count import EmojiCount
from elemeta.nlp.extractors.high_level.hashtag_count import HashtagCount
from elemeta.nlp.extractors.high_level.hinted_profanity_sentence_count import HintedProfanitySentenceCount
from elemeta.nlp.extractors.high_level.hinted_profanity_words_count import HintedProfanityWordsCount
from elemeta.nlp.extractors.high_level.link_count import LinkCount
from elemeta.nlp.extractors.high_level.mention_count import MentionCount
from elemeta.nlp.extractors.high_level.must_appear_words_percentage import MustAppearWordsPercentage
from elemeta.nlp.extractors.high_level.number_count import NumberCount
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
from elemeta.nlp.extractors.high_level.unique_word_ratio import UniqueWordRatio
from elemeta.nlp.extractors.high_level.out_of_vocabulary_count import OutOfVocabularyCount
from elemeta.nlp.extractors.high_level.word_count import WordCount
from elemeta.nlp.extractors.high_level.word_regex_matches_count import WordRegexMatchesCount

TEST_ASSET_FOLDER = os.path.join(os.path.dirname(__file__), "../assets")
TWITTER_FILE = f"{TEST_ASSET_FOLDER}/twcs.csv"
LARGE_TEXT_FILE = f"{TEST_ASSET_FOLDER}/large_text.csv"
TEXT_COLUMN = "text"

complex_metrics = [
    SentimentPolarity(),
    SentimentSubjectivity(),
    DetectLangauge(),
    EmojiCount(),
    HintedProfanityWordsCount(),
    HintedProfanitySentenceCount(),
    TextComplexity(),
]

simple_metrics = [
    UniqueWordRatio(),
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
]


@pytest.mark.parametrize(
    "name, file, text_col, metrics_list, time",
    [
        (
                "many rows. simple metrics, ~ 2 seconds per metric to go over 2092 rows",
                TWITTER_FILE,
                TEXT_COLUMN,
                simple_metrics,
                0.5,
        ),
        (
                "many rows. complex metrics",
                TWITTER_FILE,
                TEXT_COLUMN,
                complex_metrics,
                0.5,
        ),
        (
                "single row. large~ish text, simple metrics",
                LARGE_TEXT_FILE,
                TEXT_COLUMN,
                simple_metrics,
                0.5,
        ),
        (
                "single row. large~ish text, complex metrics",
                LARGE_TEXT_FILE,
                TEXT_COLUMN,
                complex_metrics,
                0.5,
        ),
    ],
)
def test_single_metrics(name, file, text_col, metrics_list, time):
    df = pandas.read_csv(file)
    for metric in metrics_list:

        start_time = datetime.now()
        met.MetadataExtractorsRunner([metric]).run_on_dataframe(df, text_col)
        end = datetime.now() - start_time
        print(name, end.seconds, metric.name)
        # assert end.seconds <= time, f"took to long {end.seconds}. expected {time} for metric {metric.name}"


@pytest.mark.parametrize(
    "name, file, text_col, metrics_list, time",
    [
        (
                "many rows. simple metrics, ~ 2 seconds per metric to go over 2092 rows",
                TWITTER_FILE,
                TEXT_COLUMN,
                simple_metrics,
                2,  # 2092 rows, 22 metrics
        ),
        (
                "many rows. complex metrics",
                TWITTER_FILE,
                TEXT_COLUMN,
                complex_metrics,
                250,
        ),
        (
                "single row. large~ish text, simple metrics",
                LARGE_TEXT_FILE,
                TEXT_COLUMN,
                simple_metrics,
                1,
        ),
        (
                "single row. large~ish text, complex metrics",
                LARGE_TEXT_FILE,
                TEXT_COLUMN,
                complex_metrics,
                2,
        ),
    ],
)
def test_metrics(name, file, text_col, metrics_list, time):
    df = pandas.read_csv(file)

    start_time = datetime.now()
    met.MetadataExtractorsRunner(metrics_list).run_on_dataframe(df, text_col)
    end = datetime.now() - start_time

    assert end.seconds <= time, f"took to long {end.seconds}. expected {time}"


# def test_single_sentence_infra():
#     df = pandas.read_csv(TWITTER_FILE)
#     metrics = met.MetricExporter(
#         df,
#         TEXT_COLUMN,
#         complex_metrics + simple_metrics,
#     )
#     start_time = datetime.now()
#     metrics.apply_single_row(2)
#     end = datetime.now() - start_time
#     assert end.seconds <= 0.5, f"took to long {end.seconds}"


def test_text_infra():
    text = "Hello, my name is Elad. What are we doing today?"
    metrics = met.MetadataExtractorsRunner(
        metadata_extractors=complex_metrics + simple_metrics,
    )
    start_time = datetime.now()
    metrics.run(text)
    end = datetime.now() - start_time
    assert end.seconds <= 0.5, f"took to long {end.seconds}"
