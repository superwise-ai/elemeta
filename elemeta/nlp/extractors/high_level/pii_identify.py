from typing import Dict, List, Optional

from presidio_analyzer import AnalyzerEngine

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import AbstractTextMetafeatureExtractor


class PII_Identify(AbstractTextMetafeatureExtractor):
    """
    Identifies any potential Named Entity Recognitions (NERs) mentioned in a text.

    Parameters
    ----------
    name : str, optional
        Name of the metafeature. If not given, it will be extracted from the class name.
    pii : list of str, optional
        List of specific Personally Identifiable Information (PII) to narrow down the analyzer.
        If none or an unsupported PII is given, the default is to search for all.
        Supported entities can be found at: https://microsoft.github.io/presidio/supported_entities/

    Examples
    --------
    >>> from elemeta.nlp.extractors.high_level.pii_identify import PII_Identify
    >>> pii = PII_Identify()
    >>> text = "My email address is john.doe@example.com and my phone number is 123-456-7890."
    >>> result = pii(text)
    >>> print(result)
    {'EMAIL_ADDRESS': ['john.doe@example.com'], 'PHONE_NUMBER': ['123-456-7890'], 'URL': ['john.do', 'example.com']}

    """

    def __init__(
        self,
        name: Optional[str] = None,
        pii: Optional[List[str]] = None,
    ):
        super().__init__(name)
        if pii is None:
            self.limit_pii = None
        else:
            valid_pii_list = [
                "CREDIT_CARD",
                "CRYPTO",
                "DATE_TIME",
                "EMAIL_ADDRESS",
                "IBAN_CODE",
                "IP_ADDRESS",
                "NRP",
                "LOCATION",
                "PERSON",
                "PHONE_NUMBER",
                "MEDICAL_LICENSE",
                "URL",
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
            if len(list(set(pii) - set(valid_pii_list))) == 0:
                self.limit_pii = pii
            else:
                self.limit_pii = None

    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Detects Named Entity Recognitions (NERs) from a text.

        Parameters
        ----------
        text : str
            The string to run the analysis on.

        Returns
        -------
        Dict[str, List[str]]
            A dictionary of the identified PII from the text, where the keys are the type of PII
            and the values are lists of all analyzed PII of that type.

        Examples
        --------
        >>> from elemeta.nlp.extractors.high_level.pii_identify import PII_Identify
        >>> extractor = PII_Identify()
        >>> text = "My email address is john.doe@example.com and my phone number is 123-456-7890."
        >>> result = extractor.extract(text)
        >>> print(result)
        {'EMAIL_ADDRESS': ['john.doe@example.com'], 'PHONE_NUMBER': ['123-456-7890'], 'URL': ['john.do', 'example.com']}

        """
        analyzer = AnalyzerEngine()
        if self.limit_pii is None:
            analyzer_results = analyzer.analyze(text=text, language="en")
        else:
            analyzer_results = analyzer.analyze(
                text=text,
                language="en",
            )
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
