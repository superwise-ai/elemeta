from elemeta.nlp.extractors.high_level.detect_langauge_langdetect import DetectLanguage
from elemeta.nlp.extractors.high_level.hinted_profanity_words_count import (
    HintedProfanityWordsCount,
)
from elemeta.nlp.extractors.high_level.injection_similarity import InjectionSimilarity
from elemeta.nlp.extractors.high_level.jailbreak_similarity import JailBreakSimilarity
from elemeta.nlp.extractors.high_level.refusal_similarity import RefusalSimilarity
from elemeta.nlp.extractors.high_level.semantic_text_pair_similarity import (
    SemanticTextPairSimilarity,
)
from elemeta.nlp.extractors.high_level.sentiment_polarity import SentimentPolarity
from elemeta.nlp.extractors.high_level.text_complexity import TextComplexity
from elemeta.nlp.extractors.high_level.text_length import TextLength
from elemeta.nlp.extractors.high_level.toxicity_extractor import ToxicityExtractor
from elemeta.nlp.runners.pair_metafeature_extractors_runner import (
    PairMetafeatureExtractorsRunner,
    PairMetafeatureExtractorsRunnerResult,
)


class CommonLLMSuite:
    def __init__(self):
        sentiment_polarity = SentimentPolarity()
        detect_language = DetectLanguage()
        text_complexity = TextComplexity()
        text_length = TextLength()
        hinted_profanity_words_count = HintedProfanityWordsCount()
        semantic_two_text_similarity = SemanticTextPairSimilarity()
        jail_break_similarity = JailBreakSimilarity()
        refusal_similarity = RefusalSimilarity()
        injection_similarity = InjectionSimilarity()
        toxicity = ToxicityExtractor()

        self.runner = PairMetafeatureExtractorsRunner(
            input_1_extractors=[
                sentiment_polarity,
                detect_language,
                text_complexity,
                text_length,
                hinted_profanity_words_count,
                toxicity,
                jail_break_similarity,
                injection_similarity,
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

    def run(self, prompt: str, llm_output: str) -> PairMetafeatureExtractorsRunnerResult:
        return self.runner.run(prompt, llm_output)
