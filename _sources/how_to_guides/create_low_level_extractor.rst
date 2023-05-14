===========================
Create Low Level Extractor
===========================
Low-level API extractor are the exporters that do not work "out of the box."
You must supply additional parameters (such as a tokenizer function).

::

    >>> from elemeta.nlp.extractors.low_level.avg_token_length import AvgTokenLength

Let’s implement 2 tokenizers:

::

    >>> my_simple_word_tokinzer = lambda text: text.split(" ")
    >>> my_simple_line_tokinzer = lambda text: text.split("\n")

Now let’s test them:

::

    >>> text = """Lorem Ipsum is simply dummy text of the printing and typesetting industry.
    ... Lorem Ipsum has been the industry's standard dummy text ever since the 1500s,
    ... when an unknown printer took a galley of type and scrambled it to make a type specimen book."""
    >>> my_simple_word_tokinzer(text)
    ['Lorem',
     'Ipsum',
     'is',
     'simply',
     'dummy',
     'text',
     'of',
     'the',
     'printing',
     'and',
     'typesetting',
     'industry.\nLorem',
     'Ipsum',
     'has',
     'been',
     'the',
     "industry's",
     'standard',
     'dummy',
     'text',
     'ever',
     'since',
     'the',
     '1500s,\nwhen',
     'an',
     'unknown',
     'printer',
     'took',
     'a',
     'galley',
     'of',
     'type',
     'and',
     'scrambled',
     'it',
     'to',
     'make',
     'a',
     'type',
     'specimen',
     'book.']
    >>> my_simple_line_tokinzer(text)
    ['Lorem Ipsum is simply dummy text of the printing and typesetting industry.',
    "Lorem Ipsum has been the industry's standard dummy text ever since the 1500s,",
    'when an unknown printer took a galley of type and scrambled it to make a type specimen book.']

We can supply those tokenizers to ``AvgTokenLength`` and create two AvgTokenLength versions

::

    >>> atl_for_words = AvgTokenLength(tokenizer=my_simple_word_tokinzer)
    >>> atl_for_lines = AvgTokenLength(tokenizer=my_simple_line_tokinzer)

Let’s test them:

::

    >>> atl_for_words(text)
    5.0
    >>> atl_for_lines(text)
    81.0


For a full working example
please use the following `Google Colab <https://colab.research.google.com/github/superwise-ai/elemeta/blob/main/docs/notebooks/create_low_level_extractor.ipynb>`_




.. toctree::
   :maxdepth: 1
   :caption: FAQ:
