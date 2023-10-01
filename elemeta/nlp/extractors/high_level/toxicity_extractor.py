from typing import Optional

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
)

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import (
    AbstractTextMetafeatureExtractor,
)


class ToxicityExtractor(AbstractTextMetafeatureExtractor):
    """
    measures toxicity of a given text
    """

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        name: Optional[str]
            name of the metadata and if not given will extract the name from the class name
        path: Optional[str]
            the path used for the model. If not given, defaults to the hugginface library
        """

        super().__init__(name)
        self.model_path = "tillschwoerer/roberta-base-finetuned-toxic-comment-detection"

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
        result = 0.0
        toxicity_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        pipeline = TextClassificationPipeline(model=model, tokenizer=toxicity_tokenizer)
        for pair in pipeline(text):
            if pair["label"] == "TOXIC":
                result = pair["score"]
            else:
                result = 1 - pair["score"]
        return result
