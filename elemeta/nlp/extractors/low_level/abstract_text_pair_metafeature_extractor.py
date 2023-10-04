import re
from abc import abstractmethod
from typing import Any, Optional

from elemeta.common.abstract_pair_metafeature_extractor import AbstractPairMetafeatureExtractor


class AbstractTextPairMetafeatureExtractor(AbstractPairMetafeatureExtractor):
    """
    This class holds a function to be run to extract the metadata value and the name
    of the metadata
    """

    def __init__(self, name: Optional[str] = None):
        """initializer for the Metric object

        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        """
        if name:
            self.name = name
        else:
            self.name = re.sub(r"(?<!^)(?=[A-Z])", "_", self.__class__.__name__).lower()

    @abstractmethod
    def extract(self, input_1: str, input_2: str) -> Any:
        """
        This function will extract the metric from the text
        Parameters
        ----------
        input_1: str
        input_2: str

        Returns
        -------
        Any
            the metadata extracted from
        """
        raise NotImplementedError

    def __call__(self, input_1: str, input_2: str):
        """
        run self.extract on the given text

        Parameters
        ----------
        input_1: str
        input_2: str

        Returns
        -------
        Any
            the metadata extracted from text
        """
        return self.extract(input_1=input_1, input_2=input_2)
