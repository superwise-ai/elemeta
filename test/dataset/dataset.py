import pytest as pytest

from elemeta.dataset.dataset import get_avengers_endgame_tweets, get_imdb_reviews,get_tweets_likes
import pytest


@pytest.mark.parametrize(
    "name, pd_f, rows",
    [
        (
            "avenger_endgame_tweets",
            get_avengers_endgame_tweets,
            15000,
        ),
        (
            "imdb_reviews",
            get_imdb_reviews,
            50000,
        ),
        (
            "get_tweets_likes",
            get_tweets_likes,
            52542,
        )


    ],
)
def test_datasets(name, pd_f, rows):
    df = pd_f()
    assert len(df) == rows, f"dataset {name} did not load well"

