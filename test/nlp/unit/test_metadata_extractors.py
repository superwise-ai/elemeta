import pytest
from nltk import TweetTokenizer, word_tokenize  # type: ignore

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
from elemeta.nlp.extractors.high_level.punctuation_count import PunctuationCount
from elemeta.nlp.extractors.high_level.regex_match_count import RegexMatchCount
from elemeta.nlp.extractors.high_level.sentence_avg_length import SentenceAvgLength
from elemeta.nlp.extractors.high_level.sentence_count import SentenceCount
from elemeta.nlp.extractors.high_level.sentiment_polarity import SentimentPolarity
from elemeta.nlp.extractors.high_level.sentiment_subjectivity import (
    SentimentSubjectivity,
)
from elemeta.nlp.extractors.high_level.special_chars_count import SpecialCharsCount
from elemeta.nlp.extractors.high_level.stop_words_count import StopWordsCount
from elemeta.nlp.extractors.high_level.syllable_count import SyllableCount
from elemeta.nlp.extractors.high_level.text_complexity import TextComplexity
from elemeta.nlp.extractors.high_level.text_length import TextLength
from elemeta.nlp.extractors.high_level.unique_word_count import UniqueWordCount
from elemeta.nlp.extractors.high_level.unique_word_ratio import UniqueWordRatio
from elemeta.nlp.extractors.high_level.out_of_vocabulary_count import (
    OutOfVocabularyCount,
)
from elemeta.nlp.extractors.high_level.word_count import WordCount
from elemeta.nlp.extractors.high_level.word_regex_matches_count import (
    WordRegexMatchesCount,
)
from elemeta.nlp.extractors import length_check_basic, avg_check_basic


# TODO for all check tokenizer difference. example can be between twitter and not. the parse isn't differently


@pytest.mark.parametrize(
    "name, text, sentiment_min, sentiment_max",
    [
        ("positive sentence", "I love cakes!", 0.25, 1),
        ("negative sentence", "I HATE cakes!", -1, -0.25),
        ("neutral sentence", "meh i dont know", -0.25, 0.25),
        (
            "many sentences",
            "It was a long day. I did alot today. I am happy, I feels good to be a live. Life is life",
            -1,
            1,
        ),
    ],
)
def test_sentiment_analysis(name, text, sentiment_min, sentiment_max):
    sentiment = SentimentPolarity().extract(text)
    assert (
        sentiment >= sentiment_min
    ), f"output {sentiment} is smaller than min threshold {sentiment_min} for test {name}"
    assert (
        sentiment <= sentiment_max
    ), f"output {sentiment} is larger than max threshold {sentiment_max} for test {name}"


@pytest.mark.parametrize(
    "name, text, sentiment_min, sentiment_max",
    [
        ("subjective sentence", "I hate cakes!", 0.5, 1),
        ("objective sentence", "They all failed the test", 0, 0.5),
    ],
)
def test_sentiment_analysis(name, text, sentiment_min, sentiment_max):
    sentiment = SentimentSubjectivity().extract(text)
    assert (
        sentiment >= sentiment_min
    ), f"output {sentiment} is smaller than min threshold {sentiment_min} for test {name}"
    assert (
        sentiment <= sentiment_max
    ), f"output {sentiment} is larger than max threshold {sentiment_max} for test {name}"


@pytest.mark.parametrize(
    "name, text, language",
    [
        ("english", "I love cakes. Its the best. almost like the rest", "en"),
        (
            "germen",
            "Ein Gedicht von Paul Celan über den Holocaust. Auf der Webseite finden Sie den Text, eine Übersetzung",
            "de",
        ),
        ("french", "ous appelez-vous?", "fr"),
        (
            "mixed",
            """
                China (simplified Chinese: 中国; traditional Chinese: 中國),
                officially the People's Republic of China (PRC), is a sovereign state
                located in East Asia.
                """,
            "en",
        ),
    ],
)
def test_langauge_detection(name, text, language):
    lan = DetectLanguage().extract(text)
    assert (
        lan == language
    ), f"output detected {lan}. should be {language} for test {name}"


@pytest.mark.parametrize(
    "name, text, expected",
    [
        ("no emoji", "I love cake. its the best. almost like the rest", 0),
        ("6 emojies", "🤔 🙈 me así, bla es se 😌 ds 💕👭👙", 6),
    ],
)
def test_emoji_counter(name, text, expected):
    emoji_num = EmojiCount().extract(text)
    assert (
        emoji_num == expected
    ), f"output detected {emoji_num}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, exception,expected",
    [
        ("no text", "", set(), 0),
        ("valid text", "I love to move it move it", set(), 3 / 5),
        ("valid text", "I love to move it move it", {"I", "it", "not"}, 2 / 3),
    ],
)
def test_unique_words_ratio(name, text, exception, expected):
    unique_num = UniqueWordRatio(exceptions=exception).extract(text)
    assert (
        unique_num == expected
    ), f"output expected {unique_num}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, exception,expected",
    [
        ("no text", "", set(), 0),
        ("valid text", "he is not he. he is me", set(), 3),
        ("valid text", "he is not he. he is me", {"he", "is", ".", "not"}, 1),
    ],
)
def test_unique_words_count(name, text, exception, expected):
    unique_num = UniqueWordCount(exceptions=exception).extract(text)
    assert (
        unique_num == expected
    ), f"output expected {unique_num}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, regex,expected",
    [
        ("matching", "he hee is", "h.+", 2),
        ("no matching", "he is", "me", 0),
    ],
)
def test_regex(name, text, regex, expected):
    reg = WordRegexMatchesCount(regex=regex).extract(text)
    assert (
        reg == expected
    ), f"output detected {reg}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, tokenizer, condition,expected",
    [
        (
            "length check twitter",
            "hello, is it me it's you are looking for",
            TweetTokenizer().tokenize,
            lambda token: len(token) < 3,
            4,
        ),
        (
            "length check nltk",
            "hello, is it me it's you are looking for",
            word_tokenize,
            lambda token: len(token) < 3,
            6,
        ),
    ],
)
def test__length_check_basic(name, text, tokenizer, condition, expected):
    function = length_check_basic(tokenizer=tokenizer, condition=condition)
    res = function(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, tokenizer, condition,expected",
    [
        (
            "length check twitter",
            "hello, is it me it's you are looking for",
            TweetTokenizer().tokenize,
            lambda token: len(token) < 3,
            1.75,
        ),
        (
            "length check nltk",
            "hello, is it me it's you are looking for",
            word_tokenize,
            lambda token: len(token) < 3,
            1.8333333333333333,
        ),
    ],
)
def test__avg_check_basic(name, text, tokenizer, condition, expected):
    function = avg_check_basic(tokenizer=tokenizer, condition=condition)
    res = function(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, expected",
    [
        ("digit count", "I am 17 years old and i will be 1 0 0 one day", 4),
        (
            "0 digit count",
            "I am seventeen years old and i will be twenty one day",
            0,
        ),
    ],
)
def test_number_count(name, text, expected):
    res = NumberCount().extract(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, existing, expected",
    [
        ("default vocabulary", "Rick said Wubba Lubba dub-dub", None, 3),
        ("Many sentences", "Rick said Wubba Lubba dub-dub. Second sentence!!", None, 3),
        ("custom vocabulary", "I am ok now", set(["i", "am"]), 2),
    ],
)
def test_unknown_words_count(name, text, existing, expected):
    if existing is None:
        res = OutOfVocabularyCount().extract(text)
    else:
        res = OutOfVocabularyCount(vocabulary=existing).extract(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, appearing, expected",
    [
        ("no appearing", "I am ok now", set("Love"), 0),
        ("some appearing", "I am ok now", {"I", "am"}, 2 / 2),
    ],
)
def test_must_appear_count(name, text, appearing, expected):
    res = MustAppearWordsPercentage(must_appear=appearing).extract(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, expected",
    [
        ("1 sentence", "One i was afraid i was petrified", 1),
        (
            "many sentences",
            "Hello, my name is Inigo Montoya. You killed my father. Prepare to die.",
            3,
        ),
    ],
)
def test_sentence_count(name, text, expected):
    res = SentenceCount().extract(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, expected",
    [
        ("1 sentence", "One i was afraid i was petrified", 32),
        (
            "many sentences",
            "Hello, my name is Inigo Montoya. You killed my father. Prepare to die.",
            22.666666666666668,
        ),
    ],
)
def test_sentence_avg(name, text, expected):
    res = SentenceAvgLength().extract(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, expected",
    [
        ("1 sentence", "One i was afraid i was petrified", 7),
        (
            "many sentences",
            "Hello, my name is Inigo Montoya. You killed my father. Prepare to die.",
            13,
        ),
    ],
)
def test_text_size(name, text, expected):
    res = WordCount().extract(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, expected",
    [
        ("1 sentence", "One i was afraid i was petrified", 3.7142857142857144),
        (
            "many sentences",
            "Hello, my name is Inigo Montoya. You killed my father. Prepare to die.",
            4.153846153846154,
        ),
    ],
)
def test_avg_word_length(name, text, expected):
    res = AvgWordLength().extract(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, expected",
    [
        ("1 sentence", "One i was afraid i was petrified", 32),
        (
            "many sentences",
            "Hello, my name is Inigo Montoya. You killed my father. Prepare to die.",
            70,
        ),
    ],
)
def test_text_length(name, text, expected):
    res = TextLength().extract(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, expected",
    [("stop words", "Once I was afraid, I was petrified", 5)],
)
def test_count_stopwords(name, text, expected):
    res = StopWordsCount().extract(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, expected",
    [("stop words", "-{!_....(:)[,\"]?;}' I was afraid, I was petrified!", 16)],
)
def test_count_punctuation(name, text, expected):
    res = PunctuationCount().extract(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, expected",
    [
        (
            "special chars",
            "!@#$%^&*<>?/\\_+-=~`[]{}.,One, <>? I was afraid, I was petrified!",
            21,
        )
    ],
)
def test_count_special_chars(name, text, expected):
    res = SpecialCharsCount().extract(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, expected",
    [("upper lower case chars", "HalF Ok", 1 / 2), ("no text", "", 0)],
)
def test_case_ratio(name, text, expected):
    res = CapitalLettersRatio().extract(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, expected",
    [
        ("no email", "@not_a_email", 0),
        (
            "2 emails",
            "lior.something@gmail.ac.il is ok but lior@superwise.il is better",
            2,
        ),
    ],
)
def test_email_count(name, text, expected):
    res = EmailCount().extract(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, expected",
    [("no link", "myname@gmail.com", 0), ("1 link", "https://google.com", 1)],
)
def test_link_count(name, text, expected):
    res = LinkCount().extract(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, expected",
    [
        ("simple", "This love cakes", 119.19),
        (
            "complex",
            "Production of biodiesel by enzymatic transesterifcation of non-edible Salvadora persica (Pilu) oil and crude coconut oil in a solvent-free system",
            17.34,
        ),
    ],
)
def test_complex_count(name, text, expected):
    res = TextComplexity().extract(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, expected",
    [
        ("no hah", "not#me2", 0),
        ("hash", "#me2", 1),
    ],
)
def test_hashtag_count(name, text, expected):
    res = HashtagCount().extract(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, expected",
    [
        ("no mention", "someone@gmail.com", 0),
        ("mention", "@elad", 1),
    ],
)
def test_mention_count(name, text, expected):
    res = MentionCount().extract(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, expected",
    [
        ("hyperemotionality", "hyperemotionality", 8),
        ("phenomenon", "phenomenon", 4),
        ("juxtaposition", "juxtaposition", 5),
        ("conundrum", "conundrum", 3),
        ("archaeology", "archaeology", 5),
    ],
)
def test_syllables_count(name, text, expected):
    res = SyllableCount().extract(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, expected",
    [
        ("acronym count", "W.T.F that was LOL", 2),
    ],
)
def test_acronym_count(name, text, expected):
    res = AcronymCount().extract(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, expected",
    [
        ("bad words", "fuck this sh!t,I want to fucking leave the country", 3),
    ],
)
def test_profanity_words_count(name, text, expected):
    res = HintedProfanityWordsCount().extract(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, expected",
    [
        (
            "bad sentences",
            "fuck this sh!t,I want to fucking leave the country, but I am fine",
            1,
        ),
    ],
)
def test_profanity_sentences_count(name, text, expected):
    res = HintedProfanitySentenceCount().extract(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, text, expected",
    [
        (
            "dates",
            "Entries are due by January 4th, 2017 at 8:00pm, created 01/15/2005 by ACME Inc. and associates.",
            2,
        ),
    ],
)
def test_date_count(name, text, expected):
    res = DateCount().extract(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"


@pytest.mark.parametrize(
    "name, regex ,text, expected",
    [
        (
            "detected_specific_word",
            "test",
            "This is a test text, will this test pass?",
            2,
        ),
        ("no_matches", "Elemeta", "This is a test text, will this test pass?", 0),
        ("no_text", "test", "", 0),
        (
            "detected_specific_word",
            "test",
            "This is a test text, will this test pass?",
            2,
        ),
    ],
)
def test_regex_match_count(name, regex, text, expected):
    res = RegexMatchCount(regex=regex).extract(text)
    assert (
        res == expected
    ), f"output detected {res}. should be {expected} for test {name}"
