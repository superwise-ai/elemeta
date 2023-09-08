from typing import Optional

from transformers import (  # PreTrainedTokenizer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
)

from elemeta.nlp.extractors.low_level.abstract_metafeature_extractor import (
    AbstractMetafeatureExtractor,
)


class ToxicityExtractor(AbstractMetafeatureExtractor):
    """
    measures toxicity of a given text
    """

    def __init__(
        self,
        name: Optional[str] = None,
        # tokenizer: Callable[[str], List[str]],
        # path: Optional[str] = None
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
        # self.tokenizer = tokenizer
        # if path is None:
        #    self.model_path = "tillschwoerer/roberta-base-finetuned-toxic-comment-detection"
        # else:
        #    self.model_path = path

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
        # toxicity_tokenizer = LambdaTokenizer(tokenizer_func= self.tokenizer)
        # if self.model_path == "martin-ha/toxic-comment-model":
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
        # else:
        # if the model path is not the same as the huggingface library
        #    return
