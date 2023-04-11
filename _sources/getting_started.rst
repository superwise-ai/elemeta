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

    from elemeta.nlp.metadata_extractor_runner import MetadataExtractorsRunner
    metadata_extractor_runner = MetadataExtractorsRunner()
    # Running on all the data should take around a minute
    tweets = metadata_extractor_runner.run_on_dataframe(dataframe = tweets,text_column="text")
    tweets.head()

.. image:: ./images/elemeta_reviews.gif
        :width: 600
        :alt: histogram of text_length feature

Pandas DataFrames
==================
Elemeta can enrich standard dataframe objects:

.. code-block:: python

    from elemeta.nlp.metadata_extractor_runner import MetadataExtractorsRunner
    import pandas as pd

    df = pd.DataFrame({"text": ["Hi I just met you, and this is crazy","What does the fox say?","I love robots"] })
    metadata_extractor_runner = MetadataExtractorsRunner()
    df_with_metadata = metadata_extractor_runner.run_on_dataframe(dataframe=df,text_column="text")
    df_with_metadata.head()


Strings
=======
Elemeta can enrich specific strings:

.. code-block:: python

    from elemeta.nlp.metadata_extractor_runner import MetadataExtractorsRunner

    metadata_extractor_runner = MetadataExtractorsRunner()
    metadata_extractor_runner.run("This is a text about how good life is :)")



.. toctree::
   :maxdepth: 1
   :caption: FAQ: