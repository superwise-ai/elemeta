========================
Custom Extractor
========================
| It is possible to create your own ``MetadataExtractor`` to fit you needs.
| You can do this by implementing the abstract class ``AbstractMetadataExtractor``

::

    >>> from elemeta.nlp.extractors.low_level.abstract_metafeature_extractor import AbstractMetafeatureExtractor

    Let’s create IsPalindromeExtractor that will return if the given text is palindrome:

Let’s create IsPalindromeExtractor that will return if the given text is palindrome:

::

    >>> class IsPalindromeExtractor(AbstractMetadataExtractor):
    ...     def extract(self, text: str) -> bool:
    ...         normalized_text = text.replace(" ", "").lower()
    ...         return normalized_text == normalized_text[::-1]
    >>> ipe = IsPlindromExtractor()

Let’s test it:

::

    >>> ipe("cat")
    False
    >>> ipe("taco cat")
    True

Now you can easily use it and add it your ``MetadataExtractorRunner``

::

    >>> from elemeta.nlp.metafeature_extractors_runner import MetafeatureExtractorsRunner
    >>> metafeature_extractors_runner = MetafeatureExtractorsRunner()
    >>> metafeature_extractors_runner.add_metafeature_extractor(ipe)
    >>> metafeature_extractors_runner.run("Never odd or even")
    {'detect_langauge': 'en',
     'emoji_count': 0,
     'text_complexity': 92.8,
     'unique_word_ratio': 1.0,
     'unique_word_count': 4,
     'word_regex_matches_count': 4,
     'number_count': 0,
     'out_of_vocabulary_count': 1,
     'must_appear_words_ratio': 0,
     'sentence_count': 1,
     'sentence_avg_length': 17.0,
     'word_count': 4,
     'avg_word_length': 3.5,
     'text_length': 17,
     'stop_words_count': 1,
     'punctuation_count': 0,
     'special_chars_count': 0,
     'capital_letters_ratio': 0.07142857142857142,
     'regex_match_count': 1,
     'email_count': 0,
     'link_count': 0,
     'hashtag_count': 0,
     'mention_count': 0,
     'syllable_count': 5,
     'acronym_count': 0,
     'date_count': 0,
     'is_palindrome_extractor': True}

    >>> metafeature_extractors_runner.run("I love cats")
    {'detect_langauge': 'ca',
     'emoji_count': 0,
     'text_complexity': 119.19,
     'unique_word_ratio': 1.0,
     'unique_word_count': 3,
     'word_regex_matches_count': 3,
     'number_count': 0,
     'out_of_vocabulary_count': 1,
     'must_appear_words_ratio': 0,
     'sentence_count': 1,
     'sentence_avg_length': 11.0,
     'word_count': 3,
     'avg_word_length': 3.0,
     'text_length': 11,
     'stop_words_count': 0,
     'punctuation_count': 0,
     'special_chars_count': 0,
     'capital_letters_ratio': 0.1111111111111111,
     'regex_match_count': 1,
     'email_count': 0,
     'link_count': 0,
     'hashtag_count': 0,
     'mention_count': 0,
     'syllable_count': 3,
     'acronym_count': 1,
     'date_count': 0,
     'is_palindrome_extractor': False}


For a full working example
please use the following `Google Colab <https://colab.research.google.com/github/superwise-ai/elemeta/blob/main/docs/notebooks/custom_extractor.ipynb>`_





.. toctree::
   :maxdepth: 1
   :caption: FAQ:
