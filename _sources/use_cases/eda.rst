EDA
=============

| Here is a simple EDA example showing how generic information extraction can be valuable. This example illustrates how extracted metafeatures contain information that can be used to predict how many likes a tweet will receive.
| Let's start by enriching our tweets dataset again:

.. code-block:: python

    from elemeta.dataset.dataset import get_tweets_likes
    tweets_eda = get_tweets_likes().sample(5000)

.. image:: ../images/eda_basic_tweets.png
        :width: 600
        :alt: the source data set

Let's start by enriching our tweets dataset

.. code-block:: python

    from elemeta.nlp.metafeature_extractors_runner import MetafeatureExtractorsRunner

    metafeature_extractors_runner = MetafeatureExtractorsRunner()
    print("The original dataset had {} columns".format(tweets_eda.shape[1]))

    # The enrichment process
    print("Processing...")
    tweets_eda = metafeature_extractors_runner.run_on_dataframe(dataframe=tweets_eda,text_column='content')
    print("The transformed dataset has {} columns".format(tweets_eda.shape[1]))

Now let's enrich the data:

Let's look at the distribution of labels (number of likes). We can clearly see a long right-tail distribution.

.. code-block:: python

    import seaborn as sns
    import matplotlib.pyplot as plt


    sns.displot(tweets_eda, x="number_of_likes",kind="kde")


.. image:: ../images/eda_number_of_links_distribution.png
        :width: 600
        :alt: histogram of text_length feature


According to the below analysis, there is a clear correlation between tweet language and likes, since number_of_likes distribute differently between languages.

.. code-block:: python

    plt.subplots(figsize=(10,7))
    sns.boxplot(x="detect_langauge", y="number_of_likes", data=tweets_eda);

.. image:: ../images/eda_number_of_link_detect_langauge.png
        :width: 600
        :alt: histogram of word_count feature


Apart from a few outliers, tweets with at least one emoji get more likes.

.. code-block:: python

    tweets_eda['has_emoji'] = tweets_eda['emoji_count'].apply(lambda x: 'False' if x <= 0 else 'True')
    plt.subplots(figsize=(10,7))
    sns.boxplot(x="has_emoji", y="number_of_likes", data=tweets_eda)

.. image:: ../images/eda_number_of_link_has_emoji.png
        :width: 600
        :alt: joint plot on number_of_positive_words,number_of_negative_words and sentiment


For a full working example
please use the following `Google Colab <https://colab.research.google.com/github/superwise-ai/elemeta/blob/main/docs/notebooks/EDA.ipynb>`_
