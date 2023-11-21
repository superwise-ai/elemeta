from typing import Dict, List, Optional

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import (
    AbstractTextMetafeatureExtractor,
)


class NER_Identifier(AbstractTextMetafeatureExtractor):
    """
    identifies any potential PII mentioned in a text
    """

    def __init__(self, name: Optional[str] = None, path: Optional[str] = None):
        """
        Parameters
        ----------
        name: Optional[str]
            name of the metadata and if not given will extract the name from the class name
        path: Optional[str]
            the path used for the model. If not given, defaults to the hugginface library:
            https://huggingface.co/dslim/bert-base-NER
        """
        super().__init__(name)
        if path is None:
            self.model_path = "dslim/bert-base-NER"
        else:
            self.model_path = path

    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        detects NER from a text

        Parameters
        ----------
        text: str
            the string to run on

        Returns
        -------
        Dict[str, List[str]]
            returns a dictionary such that keys are:
            B-MIS	Beginning of a miscellaneous entity right after another miscellaneous entity
            I-MIS	Miscellaneous entity
            B-PER	Beginning of a person’s name right after another person’s name
            I-PER	Person’s name
            B-ORG	Beginning of an organization right after another organization
            I-ORG	organization
            B-LOC	Beginning of a location right after another location
            I-LOC	Location
            and the value are associated information extracted from the text

        """
        model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        ner = pipeline("ner", model=model, tokenizer=tokenizer)
        entities = ner(text)
        result: Dict[str, List[str]] = dict()
        for entity in entities:
            if entity["entity"] in result:
                result[entity["entity"]].append(entity["word"])
            else:
                result[entity["entity"]] = [entity["word"]]
        return result
