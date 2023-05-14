========================
Supported Languages
========================
Within Elemeta, metafeatures are currently split into two groups of metrics, statistical metrics and contextual metrics. Statistical metrics calculate technical values such as word length, word count, etc., and contextual metrics extract information regarding the context of the text.
Statistical metrics are language agnostic, while contextual metrics currently support English and, to some extent, Indo-European languages (not tested).



Statistical metrics
==================================
- UniqueWordRatio
- WordRegexMatchesCount
- NumberCount
- OutOfVocabularyCount (the default vocabulary contains only English words; therefore, with the default parameter, only English is supported)
- MustAppearWordsRatio
- SentenceCount
- SentenceAvgLength
- WordCount
- AvgWordLength
- TextLength
- StopWordsCount
- PunctuationCount
- SpecialCharsCount
- CapitalLettersRatio
- RegexMatchCount
- EmailCount
- LinkCount
- HashtagCount
- MentionCount
- SyllableCount
- AcronymCount
- DateCount



Contextual metrics
==================================
- SentimentPolarity
- SentimentSubjectivity
- DetectLanguage
- EmojiCount
- HintedProfanityWordsCount
- HintedProfanitySentenceCount
- TextComplexity

.. toctree::
   :maxdepth: 1
   :caption: Supported languages:
