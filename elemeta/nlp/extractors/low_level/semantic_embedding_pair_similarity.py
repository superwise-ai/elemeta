from sentence_transformers import util
from torch import Tensor

from elemeta.common.abstract_pair_metafeature_extractor import AbstractPairMetafeatureExtractor


class SemanticEmbeddingPairSimilarity(AbstractPairMetafeatureExtractor):
    def extract(self, input_1: Tensor, input_2: Tensor) -> Tensor:
        return util.cos_sim(input_1, input_2)
