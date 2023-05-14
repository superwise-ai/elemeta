========================
Metafeatures
========================
| Within Elemeta, metafeatures are currently split into two groups of metrics, statistical metrics and contextual metrics. Statistical metrics calculate technical values such as word length, word count, etc., and contextual metrics extract information regarding the context of the text.
| Here you can find explanations on all exiting metafeatures and usage examples.
| You can see the full API reference `here <https://docs.elemeta.ai/elemeta.nlp.extractors.high_level.html>`_

UniqueWordRatio
---------------
Gives the ratio between the number of distinct words (total number of different values regardless how many times it appears in the dataset) to the number  of unique words (total number of values that only appear once in the dataset).

.. code-block:: python

    UniqueWordRatio(exceptions=exception)("I love to move it move it")

Will return 3/5 because there are 5 distinct words in the text: I, love, to, move, it. And 3 of them are unique: “I”, “love”, “to”.

WordRegexMatchesCount
---------------------
For a given regex return the number of words matching the regex.

.. code-block:: python

    WordRegexMatchesCount(regex="h.+")("he hee is",)

Will return:
2
because there are 2 words that match the regex: "he", "hee".


NumberCount
-----------
Counts the number of numbers in the text.

.. code-block:: python

    NumberCount()("I am 17 years old and I will be 100 one day")

Will return 2
because there are 2 numbers in the text: 17,100.

OutOfVocabularyCount
--------------------
For a given vocabulary (the default is English vocabulary taken from `nltk.corpus <https://www.nltk.org/api/nltk.corpus.html>`_) return the number of words outside of the vocabulary.

.. code-block:: python

    OutOfVocabularyCount().extract("Rick said Wubba Lubba dub-dub")

Will return 3 because there are 3 OOV (Out of Vocabulary) words: Wabba Lubba dub-dub.

MustAppearWordsPercentage
-------------------------
For a given set of words, return the percentage of words that appeared in the text.

.. code-block:: python

    MustAppearWordsPercentage(must_appear={"I", "am"})("I am good now")

Will return 1
because all the words in the set appeared in the text.

SentenceCount
-------------
Counts the number of sentences in the text.

.. code-block:: python

    SentenceCount()("Hello, my name is Inigo Montoya. You killed my father. Prepare to die.")

Will return 3 because there are 3 sentences in the text: “Hello, my name is Inigo Montoya.“, “You killed my father.“, “Prepare to die.“.

SentenceAvgLength
-----------------
Gives the average length of sentences in the text.

.. code-block:: python

    SentenceAvgLength()("Hello, my name is Inigo Montoya. You killed my father. Prepare to die.")

Will return 22.66668.

WordCount
---------
Gives the number of words in the text

.. code-block:: python

    WordCount()("Hello, my name is Inigo Montoya. You killed my father. Prepare to die.")

Will return 13.

AvgWordLength
-------------
Gives the average length of the words in the text.

.. code-block:: python

    AvgWordLength()("Hello, my name is Inigo Montoya. You killed my father. Prepare to die.")

Will return 4.538…

TextLength
----------
Gives the number of characters in the text (including whitespace).

.. code-block:: python

    TextLength()("Hello, my name is Inigo Montoya. You killed my father. Prepare to die.")

Will return 70.

StopWordsCount
--------------
Counts the number of stop words.

.. code-block:: python

    StopWordsCount()("Once I was afraid, I was petrified")

Will return 5 because there are 4 stop words in the text: “Once”, “I”, “was”, “I”, “was”.

PunctuationCount
----------------
Counts the number of punctuation marks in the text (the default list of punctuation marks can be found `here <list>`_).

.. code-block:: python

    PunctuationCount()("Once I was afraid, I was petrified!")

Will return 2 because there are 2 punctuation marks: “,”, “!”.

SpecialCharsCount
-----------------
Counts the number of special characters in the text (the default list of special characters can be found `here <list>`_).

.. code-block:: python

    SpecialCharsCount()("Once I was afraid, I was petrified!")

Will return 1
because there is 1 special character.

CapitalLettersRatio
-------------------
Counts the ratio of capital letters to all letters.

.. code-block:: python

    CapitalLettersRatio()("HalF Ok")

Will return 0.5.

RegexMatchCount
---------------
For a given regex return the number of matches it has in the text.

.. code-block:: python

    RegexMatchCount(regex="test")("This is a test text, will this test pass?")

Will return 2
because the regex will match on the word “test” and it appears twice.

EmailCount
----------
Counts the number of emails in the text.

.. code-block:: python

    EmailCount()("lior.something@gmail.ac.il is ok but lior@superwise.il is better")

Will return 2

LinkCount
---------
Counts the number of links in the text.

.. code-block:: python

    LinkCount()("https://google.com")

Will return 1

HashtagCount
------------
Counts the number of hashtags in the text.

.. code-block:: python

    HashtagCount()("#me2")

Will return 1.

MentionCount
-------------
Counts the number of mentions (word in the format @someones_name).

.. code-block:: python

    MentionCount()("@elad")

Will return 1.

SyllableCount
-------------
Counts the total number of syllables in the text

.. code-block:: python

    SyllableCount()("hyperemotionality")

Will return 8.

AcronymCount
------------
Counts the number of acronyms in the text.

.. code-block:: python

    AcronymCount()("W.T.F that was LOL")

Will return 2.

DateCount
---------
Counts the number of dates in the text.

.. code-block:: python

    DateCount()("Entries are due by January 4th, 2017 at 8:00pm, created 01/15/2005 by ACME Inc. and associates.")

Will return 2.

SentimentPolarity
-----------------
Returns the Sentiment Polarity (read more about the difference between sentiment polarity and sentiment subjectivity `here <https://www.tasq.ai/tasq-question/what-are-polarity-and-subjectivity-in-sentiment-analysis/>`__) value as a range between -1 to 1, where -1 means the text is an utterly negative sentiment and 1 is an utterly positive sentiment.

.. code-block:: python

    SentimentPolarity()("I love cake!")

Will return around 0.669.

.. code-block:: python

    SentimentPolarity()("I HATE cake!")

Will return around -0.693.

SentimentSubjectivity
---------------------
Returns the Sentiment Subjectivity (read more about the difference between sentiment polarity and sentiment subjectivity `here <https://www.tasq.ai/tasq-question/what-are-polarity-and-subjectivity-in-sentiment-analysis/>`__) value as a range between 0 to 1, where 0.0 is utterly objective, and 1.0 is utterly subjective.

.. code-block:: python

    SentimentSubjectivity()("I hate cakes!")

Will return around 0.9.

.. code-block:: python

    SentimentSubjectivity()("They all failed the test")

Will return around 0.3.

DetectLanguage
---------------
Returns the language of the text (uses `langdetect <https://github.com/Mimino666/langdetect>`_ library).

.. code-block:: python

    DetectLanguage()("I love cakes. Its the best. Almost like the rest")

Will return “en”.

EmojiCount
----------
Counts the number of emojis in the text.

.. code-block:: python

    EmojiCount()("🤔 🙈 me así, bla es se 😌 ds 💕👭👙")

Will return 6.

HintedProfanityWordsCount
-------------------------
Counts the number of profanity words (uses better_profanity `better_profanity <https://github.com/snguyenthanh/better_profanity>`_ library).

.. code-block:: python

    HintedProfanityWordsCount()("Fuck this sh!t. I want to fucking leave the country")

Will return 3.

HintedProfanitySentenceCount
----------------------------
Counts the number of sentences with profanity words in them (uses `better_profanity <https://github.com/snguyenthanh/better_profanity>`_ library).

.. code-block:: python

    HintedProfanitySentenceCount()("Fuck this sh!t. I want to fucking leave the country, but I am fine")

Will return 1.


TextComplexity
--------------
Return the ``Flesch Reading Ease Score`` of the text.

.. code-block:: python

    TextComplexity()("This love cakes")

Will return 119.19.

.. code-block:: python

    TextComplexity()("Production of biodiesel by enzymatic transesterifcation of non-edible Salvadora persica (Pilu) oil and crude coconut oil in a solvent-free system")

Will return 17.34.



.. toctree::
   :maxdepth: 1
   :caption: FAQ:
