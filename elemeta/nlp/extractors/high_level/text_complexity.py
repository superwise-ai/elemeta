from typing import Callable, Optional

import textstat  # type: ignore

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import AbstractTextMetafeatureExtractor


class TextComplexity(AbstractTextMetafeatureExtractor):
    """
    Return the Flesch Reading Ease Score of the text

    Example
    -------
    >>> from elemeta.nlp.extractors.high_level.text_complexity import TextComplexity
    >>> text_complexity = TextComplexity()
    >>> print(text_complexity("This love cakes"))  # Output: 119.19
    >>> print(text_complexity("Production of biodiesel by enzymatic transesterifcation of non-edible Salvadora persica (Pilu) oil and crude coconut oil in a solvent-free system"))  # Output: 17.34
    """

    def __init__(
        self,
        metric: Callable[[str], float] = textstat.textstat.flesch_reading_ease,
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        name: Optional[str]
            name to of the metadata of not given will extract the name from the class name
        metric: Callable[[str],List[str]]
            a metadata that quantifies the complexity of the text.
            Default is 'flesch reading ease' by the 'textstat' package
        """
        super().__init__(name)
        self.metric = metric

    def extract(self, text: str) -> float:
        return self.metric(text)
