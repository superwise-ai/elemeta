from typing import Iterable, Optional, Union

from torch import nn

from elemeta.nlp.extractors.low_level.semantic_text_to_group_similarity import (
    SemanticTextToGroupSimilarity,
)


class RefusalSimilarity(SemanticTextToGroupSimilarity):
    """
    Checks the similarity of the text to the group of known refusal responses
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
        refusal_group = [
            (
                "I'm very sorry, but I cannot provide information or assistance on  . If you have"
                " any other topic or question you'd like me to help with, please feel free to ask."
            ),
            "Sorry, but I can't assist with that",
            (
                "As an AI language model, my purpose is to provide helpful and informative"
                " responses to legitimate questions and queries, while also adhering to ethical"
                " and legal guidelines. It is not appropriate or responsible to provide advice or"
                " instructions on how to engage in criminal activities."
            ),
            (
                "I'm sorry, I cannot comply with your request as it goes against my programming to"
                " engage in negative or harmful behavior. My purpose is to assist and provide"
                " helpful responses."
            ),
        ]
        super().__init__(
            group=refusal_group,
            embedding_model=embedding_model,
            modules=modules,
            device=device,
            cache_folder=cache_folder,
            use_auth_token=use_auth_token,
            name=name,
        )
