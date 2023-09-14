from typing import Any, List

from pydantic import BaseModel

from elemeta.nlp.extractors.low_level.abstract_text_metafeature_extractor import (
    AbstractTextMetafeatureExtractor,
)
from elemeta.nlp.extractors.low_level.abstract_text_pair_metafeature_extractor import (
    AbstractTextPairMetafeatureExtractor,
)


class PairRunnerResult(BaseModel):
    input_1: List[Any]
    input_2: List[Any]
    input_1_and_2: List[Any]


class PairRunner:
    def __init__(
        self,
        input_1_extractors: List[AbstractTextMetafeatureExtractor],
        input_2_extractors: List[AbstractTextMetafeatureExtractor],
        input_1_and_2_extractors: List[AbstractTextPairMetafeatureExtractor],
    ):
        self.input_1_extractors = input_1_extractors
        self.input_2_extractors = input_2_extractors
        self.input_1_and_2_extractors = input_1_and_2_extractors

    def run(self, input_1: Any, input_2: Any) -> PairRunnerResult:
        return PairRunnerResult(
            input_1=[extractor(input_1) for extractor in self.input_1_extractors],
            input_2=[extractor(input_2) for extractor in self.input_2_extractors],
            input_1_and_2=[
                extractor(input_1, input_2) for extractor in self.input_1_and_2_extractors
            ],
        )
