from typing import Dict, List, Optional

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import AbstractTextMetafeatureExtractor


class NER_Identifier(AbstractTextMetafeatureExtractor):
    """
    Identifies any potential PII mentioned in a text.

    Parameters
    ----------
    name : str, optional
        Name of the metafeature. If not given, the name will be extracted from the class name.
    path : str, optional
        The path used for the model. If not given, defaults to:
        https://huggingface.co/dslim/bert-base-NER

    Attributes
    ----------
    model_path : str
        The path to the NER model.

    Methods
    -------
    extract(text)
        Detects NER from a text.

    Examples
    --------
    >>> from elemeta.nlp.extractors.high_level.ner_identifier import NER_Identifier
    >>> ner_identifier = NER_Identifier()
    >>> text = "John Doe works at ABC Corp in New York."
    >>> result = ner_identifier.extract(text)
    >>> print(result)
    {
        'B-PER': ['John'],
        'I-PER': ['Do', '##e'],
        'B-ORG': ['ABC'],
        'I-ORG': ['Corp'],
        'B-LOC': ['New'],
        'I-LOC': ['York']
    }
    """

    def __init__(self, name: Optional[str] = None, path: Optional[str] = None):
        super().__init__(name)
        if path is None:
            self.model_path = "dslim/bert-base-NER"
        else:
            self.model_path = path

    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Detects NER from a text.

        Parameters
        ----------
        text : str
            The string to run the NER on.

        Returns
        -------
        Dict[str, List[str]]
            A dictionary where the keys represent the NER tags and the values are lists of associated words.
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
