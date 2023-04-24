import os

import numpy as np
import pandas
import pytest

import elemeta.nlp.metafeature_extractors_runner as met
from elemeta.nlp.extractors.high_level.avg_word_length import AvgWordLength
from elemeta.nlp.extractors.high_level.emoji_count import EmojiCount

TEST_ASSET_FOLDER = os.path.join(os.path.dirname(__file__), "../assets")
SHORT_TEXT_FILE = f"{TEST_ASSET_FOLDER}/short_text.csv"
LONG_TEXT_FILE = f"{TEST_ASSET_FOLDER}/long_text.csv"
TEXT_COLUMN = "text"


# test no excpetion, and shape
# def test_valid_dataset_runner(name, file, text_col, metrics_list, exception):
# should run single text with given metric, without, with compute intensive or without


# def test_valid_single_text_runner(name, file, text_col, metrics_list, exception):
# should run single text with given metric, without, with compute intensive or without

def test_add_metafeature():
    metric_one = AvgWordLength()
    metrics = met.MetafeatureExtractorsRunner([metric_one])

    metric_two = EmojiCount()
    metrics.add_metafeature_extractor(metric_two)

    result = metrics.run("This is my add metafeature test")
    a = 2


# def test_metrics_validity(name, file, text_col, metrics_list, exception):
#     df = pandas.read_csv(file)
#     metrics = met.MetafeatureExtractorsRunner(metafeature_extractors=metrics_list)
#     try:
#         new_df = metrics.run_on_dataframe(df, text_col)
#         if exception:
#             raise "Should have thrown an exception"
#     except Exception as e:
#         if not exception:
#             raise e

def test_default_metric_name():
    expected_default_metric_name = "avg_word_length"
    metric = AvgWordLength()

    df = pandas.read_csv(LONG_TEXT_FILE)
    metrics = met.MetafeatureExtractorsRunner([metric])
    new_df = metrics.run_on_dataframe(df, TEXT_COLUMN)
    assert (
            expected_default_metric_name in new_df.columns
    ), f"could not find name {expected_default_metric_name} in the new_df"


def test_custom_metric_name():
    expected_metric_name = "avg word length"
    metric = AvgWordLength(name=expected_metric_name)
    df = pandas.read_csv(LONG_TEXT_FILE)
    metrics = met.MetafeatureExtractorsRunner([metric])
    new_df = metrics.run_on_dataframe(df, TEXT_COLUMN)
    assert expected_metric_name in new_df.columns, f"could not find name {expected_metric_name} in the df"
    assert (
            new_df[expected_metric_name].dtypes == np.float64
    ), "new_df was not populated properly with metric. type missmatch"


def test_non_existing_column_name():
    metric = AvgWordLength()
    df = pandas.read_csv(SHORT_TEXT_FILE)
    metrics = met.MetafeatureExtractorsRunner([metric])
    with pytest.raises(AssertionError):
        metrics.run_on_dataframe(df, "I dont exist")
