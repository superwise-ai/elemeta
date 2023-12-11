import pandas as pd

from elemeta.nlp.extractors.high_level.detect_langauge_langdetect import DetectLanguage
from elemeta.nlp.extractors.high_level.hinted_profanity_words_count import (
    HintedProfanityWordsCount,
)
from elemeta.nlp.extractors.high_level.refusal_similarity import RefusalSimilarity
from elemeta.nlp.extractors.high_level.semantic_text_pair_similarity import (
    SemanticTextPairSimilarity,
)
from elemeta.nlp.extractors.high_level.sentiment_polarity import SentimentPolarity
from elemeta.nlp.extractors.high_level.text_complexity import TextComplexity
from elemeta.nlp.extractors.high_level.text_length import TextLength
from elemeta.nlp.extractors.high_level.toxicity_extractor import ToxicityExtractor
from elemeta.nlp.runners.pair_metafeature_extractors_runner import PairMetafeatureExtractorsRunner


def _replace(x):
    return (
        x.replace("input_1_and_2", "prompt_and_output")
        .replace("input_1", "prompt")
        .replace("input_2", "output")
    )


class CommonLLMSuite:
    """
    Suite that contain common misfeatures extractors that are used in the LLM monitoring
    """

    def __init__(self):
        sentiment_polarity = SentimentPolarity()
        detect_language = DetectLanguage()
        text_complexity = TextComplexity()
        text_length = TextLength()
        hinted_profanity_words_count = HintedProfanityWordsCount()
        semantic_two_text_similarity = SemanticTextPairSimilarity()
        refusal_similarity = RefusalSimilarity()
        toxicity = ToxicityExtractor()

        self.runner = PairMetafeatureExtractorsRunner(
            input_1_extractors=[
                sentiment_polarity,
                detect_language,
                text_complexity,
                text_length,
                hinted_profanity_words_count,
                toxicity,
            ],
            input_2_extractors=[
                sentiment_polarity,
                detect_language,
                text_complexity,
                text_length,
                hinted_profanity_words_count,
                toxicity,
                refusal_similarity,
            ],
            input_1_and_2_extractors=[semantic_two_text_similarity],
        )

    def run(self, prompt: str, output: str) -> dict:
        d = self.runner.run(prompt, output).model_dump()
        df = pd.json_normalize(d, sep="_")
        df.rename(columns=_replace, inplace=True)
        return df.to_dict("records")[0]
