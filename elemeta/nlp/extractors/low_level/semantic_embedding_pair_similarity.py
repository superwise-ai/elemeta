from sentence_transformers import util
from torch import Tensor

from elemeta.common.abstract_pair_metafeature_extractor import AbstractPairMetafeatureExtractor


class SemanticEmbeddingPairSimilarity(AbstractPairMetafeatureExtractor):
    """
    Calculates the semantic embedding pair similarity between two input tensors.

    Parameters:
    ----------
        input_1 (Tensor): The first input tensor.
        input_2 (Tensor): The second input tensor.

    Returns:
    -------
        Tensor: The semantic embedding pair similarity between the two input tensors.

    Examples:
    --------
        >>> import torch
        >>> from elemeta.nlp.extractors.low_level.semantic_embedding_pair_similarity import SemanticEmbeddingPairSimilarity
        >>> input_1 = torch.tensor([1, 2, 3], dtype=torch.float)
        >>> input_2 = torch.tensor([4, 5, 6], dtype=torch.float)
        >>> extractor = SemanticEmbeddingPairSimilarity()
        >>> similarity = extractor(input_1, input_2)
        >>> print(similarity) #Output: tensor([[0.9746]])
    """

    def extract(self, input_1: Tensor, input_2: Tensor) -> Tensor:
        return util.cos_sim(input_1, input_2)
