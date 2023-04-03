import os

import pandas

ASSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def get_avengers_endgame_tweets() -> pandas.DataFrame:
    return pandas.read_csv(os.path.join(ASSETS_PATH, "tweets.csv"), encoding="cp1252")


def get_imdb_reviews() -> pandas.DataFrame:
    return pandas.read_csv(os.path.join(ASSETS_PATH, "imdb_dataset.csv"))
