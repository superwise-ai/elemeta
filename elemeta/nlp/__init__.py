r"""
requirements for the nlp pacakge
"""
import string

import nltk  # type: ignore
from nltk.corpus import stopwords  # type: ignore

# download files for common use
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("words", quiet=True)

"""
English stopwords
"""
english_stopwords = set(stopwords.words("english"))

"""
Punctuations from nltk
"""
string_punctuations = set(string.punctuation)

english_punctuations = {
    "...",
    ".",
    '"',
    "'",
    "-",
    "?",
    ",",
    "'",
    ":",
    ";",
    "_",
    "[",
    "]",
    "(",
    ")",
    "{",
    "}",
    "!",
}

"""
Set of all punctuations in english
"""
extended_punctuations = string_punctuations.union(english_punctuations)

"""
Set of all special charters
"""
special_chars = {
    "!",
    "@",
    "#",
    "$",
    "%",
    "'",
    "^",
    "&",
    "*",
    "<",
    ">",
    "?",
    "/",
    "\\",
    "_",
    "+",
    "-",
    "=",
    "~",
    "`",
    "[",
    "]",
    "{",
    "}",
    ".",
}
