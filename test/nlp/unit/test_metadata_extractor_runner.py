import os

import numpy as np
import pandas
import pytest

import elemeta.nlp.metadata_extractor_runner as met
from elemeta.nlp.extractors.high_level.avg_word_length import AvgWordLength
from elemeta.nlp.extractors.high_level.capital_letters_ratio import CapitalLettersRatio
from elemeta.nlp.extractors.high_level.detect_langauge_langdetect import DetectLangauge
from elemeta.nlp.extractors.high_level.emoji_count import EmojiCount
from elemeta.nlp.extractors.high_level.must_appear_words_percentage import MustAppearWordsPercentage
from elemeta.nlp.extractors.high_level.number_count import NumberCount
from elemeta.nlp.extractors.high_level.punctuation_count import PunctuationCount
from elemeta.nlp.extractors.high_level.sentence_avg_length import SentenceAvgLength
from elemeta.nlp.extractors.high_level.sentence_count import SentenceCount
from elemeta.nlp.extractors.high_level.sentiment_polarity import SentimentPolarity
from elemeta.nlp.extractors.high_level.special_chars_count import SpecialCharsCount
from elemeta.nlp.extractors.high_level.text_length import TextLength
from elemeta.nlp.extractors.high_level.out_of_vocabulary_count import OutOfVocabularyCount
from elemeta.nlp.extractors.high_level.word_count import WordCount

TEST_ASSET_FOLDER = os.path.join(os.path.dirname(__file__), "../assets")
TWITTER_FILE = f"{TEST_ASSET_FOLDER}/twcs.csv"
LARGE_TEXT_FILE = f"{TEST_ASSET_FOLDER}/large_text.csv"
TEXT_COLUMN = "text"


@pytest.mark.parametrize(
    "name, file, text_col, metrics_list, exception",
    [
        (
                "single row. large~ish text",
                LARGE_TEXT_FILE,
                TEXT_COLUMN,
                [
                    SentimentPolarity(),
                    DetectLangauge(),
                    EmojiCount(),
                    NumberCount(),
                    OutOfVocabularyCount(),
                    MustAppearWordsPercentage(must_appear=set()),
                    SentenceCount(),
                    SentenceAvgLength(),
                    WordCount(),
                    AvgWordLength(),
                    TextLength(),
                    PunctuationCount(),
                    SpecialCharsCount(),
                    CapitalLettersRatio(),
                ],
                False,
        ),
        (
                "naming function duplication validation",
                LARGE_TEXT_FILE,
                TEXT_COLUMN,
                [
                    CapitalLettersRatio(),
                    CapitalLettersRatio(),
                ],
                True,
        ),
    ],
)
def test_metrics_validity(name, file, text_col, metrics_list, exception):
    df = pandas.read_csv(file)
    metrics = met.MetadataExtractorsRunner(metadata_extractors=metrics_list)
    try:
        new_df = metrics.run_on_dataframe(df, text_col)
        if exception:
            raise "Should have thrown an exception"
    except Exception as e:
        if not exception:
            raise e


def test_metrics_output():
    df = pandas.read_csv(LARGE_TEXT_FILE)
    metrics = met.MetadataExtractorsRunner(
        [
            DetectLangauge(),
            CapitalLettersRatio(),
        ],
    )
    new_df = metrics.run_on_dataframe(df, TEXT_COLUMN)
    assert new_df[DetectLangauge().name].iloc[
               0] == "en", "should detect language as english"
    assert (
            0.024571 < new_df[CapitalLettersRatio().name].iloc[0] < 0.025
    ), "Capital letter ratio should be ~0.024572"


def test_text_infra():
    text = "Hello, my name is Elad. What are we doing today?"
    metrics = met.MetadataExtractorsRunner(
        metadata_extractors=[
            EmojiCount(),
            NumberCount(),
            OutOfVocabularyCount(),
            MustAppearWordsPercentage(must_appear=set()),
            SentenceCount(),
            SentenceAvgLength(),
            WordCount(),
            AvgWordLength(),
            TextLength(),
            PunctuationCount(),
            SpecialCharsCount(),
            CapitalLettersRatio(),
        ],
    )
    metric_output = metrics.run(text)
    assert metric_output == {'emoji_count': 0, 'number_count': 0, 'out_of_vocabulary_count': 4,
                             'must_appear_words_percentage': 0, 'sentence_count': 2,
                             'sentence_avg_length': 23.5, 'word_count': 10,
                             'avg_word_length': 3.6, 'text_length': 48, 'punctuation_count': 3,
                             'special_chars_count': 2,
                             'capital_letters_ratio': 0.08333333333333333}, "output not the same"


def test_custom_name():
    metric_name = "avg word length"
    metric = AvgWordLength(name=metric_name)
    df = pandas.read_csv(LARGE_TEXT_FILE)
    metrics = met.MetadataExtractorsRunner([metric])
    new_df = metrics.run_on_dataframe(df, TEXT_COLUMN)
    assert metric_name in new_df.columns, f"could not find name {metric_name} in the df"
    assert (
            new_df[metric_name].dtypes == np.float64
    ), "new_df was not populated properly with metric. type missmatch"


def test_default_name():
    default_metric_name = "avg_word_length"
    metric = AvgWordLength()
    df = pandas.read_csv(LARGE_TEXT_FILE)
    metrics = met.MetadataExtractorsRunner([metric])
    new_df = metrics.run_on_dataframe(df, TEXT_COLUMN)
    assert (
            default_metric_name in new_df.columns
    ), f"could not find name {default_metric_name} in the new_df"
    assert (
            new_df[default_metric_name].dtypes == np.float64
    ), "new_df was not populated properly with metric. type missmatch"


def test_non_passing_non_existing_metric_column_name():
    metric = AvgWordLength()
    df = pandas.read_csv(LARGE_TEXT_FILE)
    metrics = met.MetadataExtractorsRunner([metric])
    with pytest.raises(AssertionError,
                       match="The given text_column:'I dont exist' doesn't exist in the given dataframe"):
        metrics.run_on_dataframe(df, "I dont exist")
