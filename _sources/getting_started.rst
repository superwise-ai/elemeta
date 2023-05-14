========================
Getting Started
========================

Get started with Elemeta by installing the Python library via pip

.. code-block:: bash

    pip install elemeta

| Once installed, there are a few example dataframes that can be used for testing the library.
| You can find them in `elemeta.dataset.dataset`

.. code-block:: python

    from elemeta.dataset.dataset import get_avengers_endgame_tweets
    # Load existing dataframe
    tweets = get_avengers_endgame_tweets()
    tweets.head()


After you have a dataset with the text column, you can start using the library with the following Python API:

.. code-block:: python

    from elemeta.nlp.metafeature_extractors_runner import MetafeatureExtractorsRunner
    metafeature_extractors_runner = MetafeatureExtractorsRunner()
    # Running on all the data should take around a minute
    tweets = metafeature_extractors_runner.run_on_dataframe(dataframe = tweets,text_column="text")
    tweets.head()


Pandas DataFrames
==================
Elemeta can enrich standard dataframe objects:

.. code-block:: python

    from elemeta.nlp.metafeature_extractors_runner import MetafeatureExtractorsRunner
    import pandas as pd

    df = pd.DataFrame({"text": ["Hi I just met you, and this is crazy","What does the fox say?","I love robots"] })
    metafeature_extractors_runner = MetafeatureExtractorsRunner()
    df_with_metafeatures = metafeature_extractors_runner.run_on_dataframe(dataframe=df,text_column="text")
    df_with_metafeatures.head()


Strings
=======
Elemeta can enrich specific strings:

.. code-block:: python

    from elemeta.nlp.metafeature_extractors_runner import MetafeatureExtractorsRunner

    metafeature_extractors_runner = MetafeatureExtractorsRunner()
    metafeature_extractors_runner.run("This is a text about how good life is :)")

To quickly try Elemeta please use our `quickstart colab <https://colab.research.google.com/github/superwise-ai/elemeta/blob/main/docs/notebooks/quick_start.ipynb>`_

.. toctree::
   :maxdepth: 1
   :caption: FAQ:
