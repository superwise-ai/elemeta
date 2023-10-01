from typing import Iterable, Optional, Union

from sentence_transformers import util
from torch import nn

from elemeta.nlp.extractors.high_level.embedding import Embedding
from elemeta.nlp.extractors.low_level.abstract_text_pair_metafeature_extractor import (
    AbstractTextPairMetafeatureExtractor,
)


class SemanticTextPairSimilarity(AbstractTextPairMetafeatureExtractor):
    """
    Checks the similarity of two texts
    """

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        modules: Optional[Iterable[nn.Module]] = None,
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        use_auth_token: Union[bool, str, None] = None,
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        embedding_model : Optional[str]
            The name of the SentenceTransformer model to use, by default "all-MiniLM-L6-v2"
        modules: Optional[Iterable[nn.Module]]
            This parameter can be used to create custom SentenceTransformer models from scratch.
        device: Optional[str]
            Device (like 'cuda' / 'cpu') that should be used for computation.
            If None, checks if a GPU can be used.
        cache_folder: Optional[str]
            Path to store models
        use_auth_token: Union[bool, str, None]
            HuggingFace authentication token to download private models.
        name: Optional[str]
            Name of the extractor
        """
        if embedding_model is not None:
            self.embedding_extractor = Embedding(
                embedding_model=embedding_model,
                device=device,
                cache_folder=cache_folder,
                use_auth_token=use_auth_token,
            )

        else:
            if modules is None:
                embedding_model = "all-MiniLM-L6-v2"
            else:
                embedding_model = None

            self.embedding_extractor = Embedding(
                embedding_model=embedding_model,
                modules=modules,
                device=device,
                cache_folder=cache_folder,
                use_auth_token=use_auth_token,
            )
        super().__init__(name=name)

    def extract(self, input_1: str, input_2: str) -> float:
        """
        Extracts the similarity between two texts

        Parameters
        ----------
        input_1: str
            first text
        input_2: str
            second text

        Returns
        -------
        float
            similarity between the two texts
        """
        embeddings = self.embedding_extractor.extract([input_1, input_2])
        return util.cos_sim(embeddings, embeddings)[0][1].tolist()
