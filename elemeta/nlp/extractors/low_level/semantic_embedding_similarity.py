from sentence_transformers import util
from torch import Tensor

from common.abstract_pair_metafeature_extractor import AbstractPairMetafeatureExtractor


class SemanticEmbeddingPairSimilarity(AbstractPairMetafeatureExtractor):
    def extract(self, embeddings1: Tensor, embeddings2: Tensor) -> Tensor:
        return util.cos_sim(embeddings1, embeddings2)
