import os

import pandas as pd
import pytest

import elemeta.nlp.metadata_extractor_runner as met
from elemeta.nlp.extractors.high_level.detect_langauge_langdetect import DetectLangauge
from elemeta.nlp.extractors.high_level.hinted_profanity_words_count import \
    HintedProfanityWordsCount
from elemeta.nlp.extractors.high_level.sentiment_polarity import SentimentPolarity
from elemeta.nlp.extractors.high_level.sentiment_subjectivity import SentimentSubjectivity
from elemeta.nlp.extractors.high_level.text_complexity import TextComplexity

TEST_ASSET_FOLDER = os.path.join(os.path.dirname(__file__), "../assets")


@pytest.mark.parametrize(
    "name, file, expect",
    [
        (
                "positive examples",
                "pos",
                1,
        ),
        (
                "negative examples",
                "neg",
                0,
        )
    ],
)
def test_sentiment_polarity(name, file, expect):
    results = []
    directory = os.path.join(TEST_ASSET_FOLDER, "sentiment_polarity", "txt_sentoken", file)
    for filename in os.listdir(directory):
        full_file_name = os.path.join(directory, filename)
        with open(full_file_name, "r") as f:
            data = f.read()
        metrics = met.MetadataExtractorsRunner(metadata_extractors=[SentimentPolarity()])
        enc = metrics.run(data)["sentiment_polarity"]
        results.append(round(enc, 0))
    res = round(sum(results) / len(results))
    assert res == expect, f"classified as {res} and should be {expect}"


@pytest.mark.parametrize(
    "name, file, encode, expect",
    [
        (
                "obj examples",
                "plot.tok.gt9.5000",
                "utf-8",
                0,
        ),
        (
                "sub examples",
                "quote.tok.gt9.5000",
                "ISO-8859-1",
                1,
        )
    ],
)
def test_sentiment_subjectivity(name, file, encode, expect):
    full_file_name = os.path.join(TEST_ASSET_FOLDER, "subjectivity", file)
    with open(full_file_name, "r", encoding=encode) as f:
        data = f.readlines()

    resulst = []
    metrics = met.MetadataExtractorsRunner(metadata_extractors=[SentimentSubjectivity()])
    for line in data:
        enc = metrics.run(line)["sentiment_subjectivity"]
        resulst.append(round(enc, 0))

    res = round(sum(resulst) / len(resulst))
    assert res == expect, f"classified as {res} and should be {expect}"


@pytest.mark.parametrize(
    "name, file",
    [
        (
                "language detection",
                "Language Detection.csv",
        )
    ],
)
def test_language_detection(name, file):
    full_file_name = os.path.join(TEST_ASSET_FOLDER, "language", file)
    df = pd.read_csv(full_file_name)
    language_map = {
        "English": "en",
        "Malayalam": "ml",
        "Hindi": "hi",
        "Tamil": "ta",
        "Dutch": "nl",
        "Kannada": "kn",
        "Portugeese": "pt",
        "French": "fr",
        "Spanish": "sp",
        "Greek": "el",
        "Russian": "ru",
        "Danish": "da",
        "Italian": "it",
        "Turkish": "tr",
        "Sweedish": "sv",
        "Arabic": "ar",
        "German": "de"
    }

    metrics = met.MetadataExtractorsRunner(metadata_extractors=[DetectLangauge()])
    hits = []
    for line in df.iloc:
        res = metrics.run(line["Text"])["detect_langauge"]
        exp = language_map.get(line["Language"])
        hits.append(exp == res)
    ratio = sum(hits) / len(hits)
    assert ratio > 0.88, f"classified success {res} and should be more than {88}%"


@pytest.mark.parametrize(
    "name, file",
    [
        (
                "google profanity list",
                "profanity.txt",
        )
    ],
)
def test_profanity_detection(name, file):
    full_file_name = os.path.join(TEST_ASSET_FOLDER, "profanity", file)
    with open(full_file_name, "r") as f:
        lines = f.readlines()

    metrics = met.MetadataExtractorsRunner(metadata_extractors=[HintedProfanityWordsCount()])
    hits = []
    for line in lines:
        res = metrics.run(line)["hinted_profanity_words_count"]
        hits.append(res != 1)

    ratio = sum(hits) / len(hits)
    assert ratio > 0.40, f"classified as {ratio} and should be {40}%"


@pytest.mark.parametrize(
    "name, file, expected",
    [
        (
                "difficulty amazon from https://github.com/nishkalavallabhi/OneStopEnglishCorpus/tree/master/Texts-Together-OneCSVperFile",
                "long.txt",
                66.17,
        )
    ],
)
def test_difficulty(name, file, expected):
    full_file_name = os.path.join(TEST_ASSET_FOLDER, "difficulty", file)
    with open(full_file_name, "r") as f:
        data = f.read()

    metrics = met.MetadataExtractorsRunner(metadata_extractors=[TextComplexity()])
    res = metrics.run(data)["text_complexity"]
    assert res == expected, f"classified as {res} and should be {expected}"
