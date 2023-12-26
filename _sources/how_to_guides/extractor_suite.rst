========================
Suite of Extractors
========================

If you want to calculate many metafeature values at once, use the MetafeatureExtractorsRunner.
You can either give it a list of specific extractors you want to use, or just let it use the
default ones.


-----------------------------
Personalized Extractor Runner
-----------------------------
You can give a list of the metafeature extractors you want to use and then use the
``run(text:str) -> Dict[str, Union[str, float, int]]`` function.
This function will return a dictionary where each metafeature's name is matched with its value.


::

    >>> from elemeta.nlp.runners.metafeature_extractors_runner import MetafeatureExtractorsRunner
    >>> metafeature_extractors_runner = MetafeatureExtractorsRunner(metafeature_extractors=[sp,ld])
    >>> metafeature_extractors_runner.run("This is a text about how good life is :)")
    {'sentiment_polarity': 0.7096, 'detect_language': 'en'}
Output: ``{'sentiment_polarity': 0.7096, 'detect_langauge': 'en'}``

------------------------
Default Extractor Runner
------------------------
If you don't specify any metafeature extractors, the system will use a set of default ones.

::

    >>> from elemeta.nlp.runners.metafeature_extractors_runner import MetafeatureExtractorsRunner
    >>> metafeature_extractors_runner = MetafeatureExtractorsRunner()
    >>> metafeature_extractors_runner.run("This is a text about how good life is :)")
    {'detect_language': 'en',
     'emoji_count': 0,
     'text_complexity': 113.1,
     'unique_word_ratio': 0.875,
     'unique_word_count': 7,
     'word_regex_matches_count': 11,
     'number_count': 0,
     'out_of_vocabulary_count': 2,
     'must_appear_words_ratio': 0,
     'sentence_count': 1,
     'sentence_avg_length': 40.0,
     'word_count': 9,
     'avg_word_length': 3.2222222222222223,
     'text_length': 40,
     'stop_words_count': 5,
     'punctuation_count': 2,
     'special_chars_count': 0,
     'capital_letters_ratio': 0.034482758620689655,
     'regex_match_count': 1,
     'email_count': 0,
     'link_count': 0,
     'hashtag_count': 0,
     'mention_count': 0,
     'syllable_count': 9,
     'acronym_count': 0,
     'date_count': 0}

--------------------------------------------------------------------------
Adding a new MetadataExtractor to an existing MetafeatureExtractorsRunner
--------------------------------------------------------------------------
To use a new MetadataExtractor with an existing MetafeatureExtractorsRunner, use the ``add_metafeature_extractor`` method.

::

    >>> from elemeta.nlp.extractors.high_level.regex_match_count import RegexMatchCount
    >>> number_of_good_in_text = RegexMatchCount(name="number_of_good_in_text",regex="good|Good")
    >>> metafeature_extractors_runner.add_metafeature_extractor(number_of_good_in_text)
    >>> metafeature_extractors_runner.run("This is a text about how good life is :)")
    {'detect_language': 'en',
     'emoji_count': 0,
     'text_complexity': 113.1,
     'unique_word_ratio': 0.875,
     'unique_word_count': 7,
     'word_regex_matches_count': 11,
     'number_count': 0,
     'out_of_vocabulary_count': 2,
     'must_appear_words_ratio': 0,
     'sentence_count': 1,
     'sentence_avg_length': 40.0,
     'word_count': 9,
     'avg_word_length': 3.2222222222222223,
     'text_length': 40,
     'stop_words_count': 5,
     'punctuation_count': 2,
     'special_chars_count': 0,
     'capital_letters_ratio': 0.034482758620689655,
     'regex_match_count': 1,
     'email_count': 0,
     'link_count': 0,
     'hashtag_count': 0,
     'mention_count': 0,
     'syllable_count': 9,
     'acronym_count': 0,
     'date_count': 0,
     'number_of_good_in_text': 1}

--------------------------------------------
Running the extractors on a Pandas Dataframe
--------------------------------------------
To run the extractors on a specified text column in a pandas dataframe, use ``run_on_dataframe
(dataframe: DataFrame,text_column: str)-> DataFrame``
This function will return a new dataframe with the metafeatures values as new columns.

::

    >>> from elemeta.dataset.dataset import get_imdb_reviews
    >>> reviews = get_imdb_reviews()[:200]
    >>> print("The original dataset had {} columns".format(reviews.shape[1]))
    The original dataset had 2 columns
    >>> reviews = metafeature_extractors_runner.run_on_dataframe(dataframe=reviews,text_column='review')
    >>> print("The transformed dataset has {} columns".format(reviews.shape[1]))
    The transformed dataset has 29 columns
    >>>reviews

.. image:: ../images/elemeta_reviews.gif
        :width: 600
        :alt: histogram of text_length feature







.. toctree::
   :maxdepth: 1
   :caption: FAQ:
