========================
Single Extractor
========================
For the following code examples, Elemeta assumes a single metafeature extractor.

Sentiment Polarity
===================
This extractor will automatically detect the sentiment of the text.

- Polarity ranges between [-1,1].
- -1 defines a negative sentiment, and 1 defines a positive sentiment.
- Negation words reverse the polarity.

::

    >>> from elemeta.nlp.extractors.high_level.sentiment_polarity import SentimentPolarity
    >>> sp = SentimentPolarity()
    >>> sp("I love Superwise") #Output: 0.6369
    >>> sp("I hate haters") #Output: -0.7845
    >>> sp("This is not a super happy excited sentence") #Output: -0.5337

Detect Language
================
This extractor will automatically detect the language of the text.

::

    >>> from elemeta.nlp.extractors.high_level.detect_language_fastText import DetectLanguage
    >>> ld = DetectLanguage()
    >>> ld("This text is in English") #Output: en
    >>> ld("הטקסט הזה בעברית") #Output: he
    >>> ld("Ce texte est en français") #Output: fr
    >>> ld("这段文字是法语") #Output: zh

.. toctree::
   :maxdepth: 1
   :caption: FAQ:
