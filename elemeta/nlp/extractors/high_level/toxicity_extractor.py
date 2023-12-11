from functools import partial
from typing import Callable, List, Optional

from nltk.tokenize import sent_tokenize
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
)

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import (
    AbstractTextMetafeatureExtractor,
)

DEFAULT_TOKENIZER = partial(sent_tokenize, language="english")


def _is_toxic(label: str) -> bool:
    return label == "toxic"


class ToxicityExtractor(AbstractTextMetafeatureExtractor):
    """
    measures toxicity of a given text
    """

    def __init__(
        self,
        name: Optional[str] = None,
        tokenizer: Callable = DEFAULT_TOKENIZER,
        aggregate: Callable = min,
    ):
        """
        Parameters
        ----------
        name: Optional[str]
            name of the metadata and if not given will extract the name from the class name
        path: Optional[str]
            the path used for the model. If not given, defaults to the hugginface library
        tokenizer: Optional[callable]
            the tokenizer to use. If not given, defaults to nltk.sent_tokenize.
            the model that is used has a restriction of 512 tokens,
            this method will split the text into chunks
        aggregate: Optional[callable]
            the aggregation function to use. If not given, defaults to min.
            the model that is used has a restriction
            of 512 tokens, this method will split the text into
            sentences and run the model on each sentence.
            the min aggregation is used to get the most toxic sentence.
        """

        super().__init__(name)
        self.model_path = "s-nlp/roberta_toxicity_classifier"
        self.tokenizer = tokenizer
        self.aggregate = aggregate

    def extract(self, text: str) -> float:
        """
        returns a float representing how toxic a piece of text is

        Parameters
        ----------
        text: str
            the string to run on
        Returns
        -------
        float
            a float closer to one is more toxic, closer to zero is non toxic.
        """
        toxicity_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        pipeline = TextClassificationPipeline(model=model, tokenizer=toxicity_tokenizer)
        text = self.tokenizer(text)
        labels_and_scores = pipeline(text)

        results: List[float] = [
            label_and_score["score"]
            if _is_toxic(label_and_score["label"])
            else 1 - label_and_score["score"]
            for label_and_score in labels_and_scores
        ]
        return self.aggregate(results)
