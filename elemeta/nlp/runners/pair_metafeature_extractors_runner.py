from typing import Any, Dict, List

from pydantic import BaseModel

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import (
    AbstractTextMetafeatureExtractor,
)
from elemeta.nlp.extractors.low_level.abstract_text_pair_metafeature_extractor import (
    AbstractTextPairMetafeatureExtractor,
)


class PairMetafeatureExtractorsRunnerResult(BaseModel):
    input_1: Dict[str, Any]
    input_2: Dict[str, Any]
    input_1_and_2: Dict[str, Any]


class PairMetafeatureExtractorsRunner:
    def __init__(
        self,
        input_1_extractors: List[AbstractTextMetafeatureExtractor],
        input_2_extractors: List[AbstractTextMetafeatureExtractor],
        input_1_and_2_extractors: List[AbstractTextPairMetafeatureExtractor],
    ):
        self.input_1_extractors = input_1_extractors
        self.input_2_extractors = input_2_extractors
        self.input_1_and_2_extractors = input_1_and_2_extractors

    def run(self, input_1: str, input_2: str) -> PairMetafeatureExtractorsRunnerResult:
        """
        run input_1_extractors on input_1, input_2_extractors on input_2 and
        input_1_and_2_extractors on the pair of input_1 and input_2

        Parameters
        ----------
        input_1: str
        input_2: str

        Returns
        -------
        PairMetafeatureExtractorsRunnerResult
            the metafeatures extracted from text
        """
        return PairMetafeatureExtractorsRunnerResult(
            input_1={extractor.name: extractor(input_1) for extractor in self.input_1_extractors},
            input_2={extractor.name: extractor(input_2) for extractor in self.input_2_extractors},
            input_1_and_2={
                extractor.name: extractor(input_1, input_2)
                for extractor in self.input_1_and_2_extractors
            },
        )
