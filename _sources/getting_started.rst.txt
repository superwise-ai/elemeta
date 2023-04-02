========================
Getting Started
========================

Get started with Elemeta by installing the Python library via pip

.. code-block:: bash

    pip install elemeta

| Once installed, there are a few example dataframes that can be used for testing the library.
| You can find them in `elemeta.dataset.dataset`

.. code-block:: python

    from elemeta.dataset.dataset import get_imdb_reviews
    # Load existing dataframe
    reviews = get_imdb_reviews()

After you have a dataset with the text column, you can start using the library with the following Python API:

.. code-block:: python

    from elemeta.nlp.metadata_extractor_runner import MetadataExtractorsRunner
    metadata_extractor_runner = MetadataExtractorsRunner()
    reviews = metadata_extractor_runner.run_on_dataframe(dataframe=reviews,text_column='review')
    reviews.show()

.. image:: https://s10.gifyu.com/images/ezgif.com-crop-1-1.gif
        :width: 600
        :alt: histogram of text_length feature

Pandas DataFrames
==================
Elemeta can enrich standard dataframe objects:

.. code-block:: python

    from elemeta.nlp.metadata_extractor_runner import MetadataExtractorsRunner
    Import pandas as pd

    df = pd.dataframe({"text": ["Hi I just met you, and this is crazy","What does the fox say?","I love robots" })
    metadata_extractor_runner = MetadataExtractorsRunner()
    df_with_metadata = metadata_extractor_runner.run_on_dataframe(dataframe=reviews,text_column="text")


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