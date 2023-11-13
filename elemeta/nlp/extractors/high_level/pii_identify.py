import os
from typing import Dict, List, Optional

from presidio_analyzer import AnalyzerEngine

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import (
    AbstractTextMetafeatureExtractor,
)

os.system("spacy download en_core_web_lg")


class PII_Identify(AbstractTextMetafeatureExtractor):
    """
    identifies any potential PII mentioned in a text
    """

    def __init__(
        self,
        name: Optional[str] = None,
        pii: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        name: Optional[str]
            name of the metadata and if not given will extract the name from the class name
        pii: Optional[List[str]]
            the list of specific pii to narrow down the analyzer. If none or an unsupported pii
            is given, the default is to search for all. Can only be selected from this list of
            supported entities: https://microsoft.github.io/presidio/supported_entities/
        """
        super().__init__(name)
        valid_pii_list = [
            "CREDIT_CARD",
            "CRYPTO",
            "EMAIL_ADDRESS",
            "IBAN_CODE",
            "IP_ADDRESS",
            "NRP",
            "PERSON",
            "PHONE_NUMBER",
            "MEDICAL_LICENSE",
            "US_BANK_NUMBER",
            "US_DRIVER_LICENSE",
            "US_ITIN",
            "US_PASSPORT",
            "US_SSN",
            "UK_NHS",
            "ES_NIF",
            "IT_FISCAL_CODE",
            "IT_DRIVER_LICENSE",
            "IT_VAT_CODE",
            "IT_PASSPORT",
            "IT_IDENTITY_CARD",
            "SG_NRIC_FIN",
            "AU_ABN",
            "AU_ACN",
            "AU_TFN",
            "AU_MEDICARE",
        ]
        if pii is None:
            self.limit_pii = valid_pii_list
        else:
            if len(list(set(pii) - set(valid_pii_list))) == 0:
                self.limit_pii = list(set(valid_pii_list) - set(pii))
            else:
                self.limit_pii = valid_pii_list

    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        detects PII from a text

        Parameters
        ----------
        text: str
            the string to run on

        Returns
        -------
        Dict[str, List[str]]
            returns a dictionary of the identified PII from the text such that the keys
            are the type of PII and the value is a list of all analyzed PII of that type
        """
        analyzer = AnalyzerEngine()
        analyzer_results = analyzer.analyze(text=text, language="en", entities=self.limit_pii)
        result: Dict[str, List[str]] = dict()
        for item in analyzer_results:
            start_index = item.start
            end_index = item.end
            pii = text[start_index:end_index]
            if item.entity_type in result:
                result[item.entity_type].append(pii)
            else:
                result[item.entity_type] = [pii]
        return result
