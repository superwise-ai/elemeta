import os

import numpy as np
import pandas
import pytest

import elemeta.nlp.metafeature_extractors_runner as met
from elemeta.nlp.extractors.high_level.avg_word_length import AvgWordLength
from elemeta.nlp.extractors.high_level.emoji_count import EmojiCount
from elemeta.nlp.metafeature_extractors_runner import non_intensive_metrics, intensive_metrics

TEST_ASSET_FOLDER = os.path.join(os.path.dirname(__file__), "../assets")
TEXT_FILE = f"{TEST_ASSET_FOLDER}/short_text.csv"
TEXT_COLUMN = "text"


@pytest.mark.parametrize("compute_intensive_test", [(False), (True)])
def test_valid_dataset_runner(compute_intensive_test):
    metrics = met.MetafeatureExtractorsRunner(compute_intensive=compute_intensive_test)
    df = pandas.read_csv(TEXT_FILE)
    result = metrics.run_on_dataframe(df, TEXT_COLUMN)

    assert len(result.columns) == len(non_intensive_metrics) + len(df.columns) + (
        len(intensive_metrics) if compute_intensive_test else 0), "Did not receive the expected amount of metafeatures"


@pytest.mark.parametrize("compute_intensive_test", [(False), (True)])
def test_valid_single_text_runner(compute_intensive_test):
    metrics = met.MetafeatureExtractorsRunner(compute_intensive=compute_intensive_test)
    result = metrics.run("Let's see how many features I get")
    assert len(result) == len(non_intensive_metrics) + (
        len(intensive_metrics) if compute_intensive_test else 0), "Did not receive the expected amount of metafeatures"


def test_add_metafeature():
    metric_one = AvgWordLength()
    metrics = met.MetafeatureExtractorsRunner([metric_one])

    metric_two = EmojiCount()
    metrics.add_metafeature_extractor(metric_two)

    result = metrics.run("This is my add metafeature test")
    assert len(result) == 2, "Expecting to see two metafeatures"


def test_default_metric_name():
    expected_default_metric_name = "avg_word_length"
    metric = AvgWordLength()

    df = pandas.read_csv(TEXT_FILE)
    metrics = met.MetafeatureExtractorsRunner([metric])
    new_df = metrics.run_on_dataframe(df, TEXT_COLUMN)
    assert (
            expected_default_metric_name in new_df.columns
    ), f"could not find name {expected_default_metric_name} in the new_df"


def test_custom_metric_name():
    expected_metric_name = "avg word length"
    metric = AvgWordLength(name=expected_metric_name)
    df = pandas.read_csv(TEXT_FILE)
    metrics = met.MetafeatureExtractorsRunner([metric])
    new_df = metrics.run_on_dataframe(df, TEXT_COLUMN)
    assert expected_metric_name in new_df.columns, f"could not find name {expected_metric_name} in the df"
    assert (
            new_df[expected_metric_name].dtypes == np.float64
    ), "new_df was not populated properly with metric. type missmatch"


def test_non_existing_column_name():
    metric = AvgWordLength()
    df = pandas.read_csv(TEXT_FILE)
    metrics = met.MetafeatureExtractorsRunner([metric])
    with pytest.raises(AssertionError):
        metrics.run_on_dataframe(df, "I dont exist")
