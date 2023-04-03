import pytest as pytest

from elemeta.dataset.dataset import get_avengers_endgame_tweets, get_imdb_reviews
import pytest


@pytest.mark.parametrize(
    "name, pd_f, rows",
    [
        (
            "avenger tweets pandas loading test. check size",
            get_avengers_endgame_tweets,
            15000,
        ),
        (
            "imdb pandas loading test. check size",
            get_imdb_reviews,
            50000,
        ),

    ],
)
def test_datasets(name, pd_f, rows):
    df = pd_f()
    assert len(df) == rows, f"dataset {name} did not load well"

