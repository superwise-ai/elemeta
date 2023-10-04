from elemeta.suites.common_llm_suite import CommonLLMSuite


def test_CommonLLMSuite_sanity():
    common_llm_suite = CommonLLMSuite()
    result = common_llm_suite.run(
        "Question, what kind of bear is best?", "Sorry I can't answer that question"
    )
    assert len(result.input_1) == 8
    assert len(result.input_2) == 7
    assert len(result.input_1_and_2) == 1
