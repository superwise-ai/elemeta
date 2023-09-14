import re
from abc import ABC, abstractmethod
from typing import Any, Optional


class AbstractMetafeatureExtractor(ABC):
    """
    Representation of a MetafeatureExtractor
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
    def extract(self, input: Any) -> Any:
        """
        This function will extract the metric from the text
        Parameters
        ----------
        input: str

        Returns
        -------
        Any
            the metadata extracted from text
        """
        raise NotImplementedError

    def __call__(self, input: str):
        """
        run self.extract on the given text

        Parameters
        ----------
        text: str

        Returns
        -------
        Any
            the metadata extracted from text
        """
        return self.extract(input)
