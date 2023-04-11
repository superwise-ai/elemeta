========================
Extractor Suite
========================
To extract multiple metadata values at once use MetadataExtractorsRunner. Supply a list of metadata extractors you want to run, apply a runner to the text, and get the list of metadata values. To run all the extractors on a text, use the runner function ``run(text: str) -> Dict[str, Union[str, float, int]]``

::

    >>> from elemeta.nlp.metadata_extractor_runner import MetadataExtractorsRunner
    >>> metadata_extractor_runner = MetadataExtractorsRunner(metadata_extractors=[sp,ld])
    >>> metadata_extractor_runner.run("This is a text about how good life is :)")
    {'sentiment_polarity': 0.7096, 'detect_langauge': 'en'}

If no metadata extractors are supplied, a default set of extractors will be selected.

::

    >>> from elemeta.nlp.metadata_extractor_runner import MetadataExtractorsRunner
    >>> metadata_extractor_runner = MetadataExtractorsRunner()
    >>> metadata_extractor_runner.run("This is a text about how good life is :)")
    {'detect_langauge': 'en',
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

To add a new MetadataExtractor to an existing MetadataExtractorsRunner we can use ``add_metadata_extractor(metadata_extractor: AbstractMetadataExtractor) -> None``:

::

    >>> from elemeta.nlp.extractors.high_level.regex_match_count import RegexMatchCount
    >>> number_of_good_in_text = RegexMatchCount(name="number_of_good_in_text",regex="good|Good")
    >>> metadata_extractor_runner.add_metadata_extractor(number_of_good_in_text)
    >>> metadata_extractor_runner.run("This is a text about how good life is :)")
    {'detect_langauge': 'en',
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

To run the extractors on all the dataframe columns, use ``run_on_dataframe(dataframe: DataFrame, text_column: str) -> DataFrame`` this function supplies a dataframe and the name of the text column. The function will return a new dataframe with all the metadata values as new columns.

::

    >>> from elemeta.dataset.dataset import get_imdb_reviews
    >>> reviews = get_imdb_reviews()[:200]
    >>> print("The original dataset had {} columns".format(reviews.shape[1]))
    The original dataset had 2 columns
    >>> reviews = metadata_extractor_runner.run_on_dataframe(dataframe=reviews,text_column='review')
    >>> print("The transformed dataset has {} columns".format(reviews.shape[1]))
    The transformed dataset has 29 columns
    >>>reviews

.. image:: ../images/elemeta_reviews.gif
        :width: 600
        :alt: histogram of text_length feature







.. toctree::
   :maxdepth: 1
   :caption: FAQ: