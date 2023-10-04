from typing import Iterable, List, Optional, Union

from numpy import ndarray
from sentence_transformers import SentenceTransformer
from torch import Tensor, nn

from elemeta.common.abstract_metafeature_extractor import AbstractMetafeatureExtractor


class Embedding(AbstractMetafeatureExtractor):
    """
    Extracts embeddings from a text using a SentenceTransformer model.
    """

    def __init__(
        self,
        embedding_model: Optional[str] = "all-MiniLM-L6-v2",
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
        self.model = SentenceTransformer(
            model_name_or_path=embedding_model,
            modules=modules,
            device=device,
            cache_folder=cache_folder,
            use_auth_token=use_auth_token,
        )
        super().__init__(name=name)

    """
    Extracts embeddings from a text using a SentenceTransformer model.
    """

    def extract(
        self,
        input: Union[str, List[str]],
        convert_to_tensor: bool = True,
    ) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Extracts embeddings from a text using a SentenceTransformer model.

        Parameters
        ----------
        input: Union[str, List[str]]
            text or list of texts to extract embeddings from
        convert_to_tensor: bool
            whether to convert the output to a tensor or keep it as a numpy array

        Returns
        -------
        Union[List[Tensor], ndarray, Tensor]
            embeddings of the input text(s)
        """
        return self.model.encode(input, convert_to_tensor=convert_to_tensor)
