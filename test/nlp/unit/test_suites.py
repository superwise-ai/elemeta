from elemeta.suites.common_llm_suite import CommonLLMSuite


def test_CommonLLMSuite_sanity():
    common_llm_suite = CommonLLMSuite()
    result = common_llm_suite.run(
        "Question, what kind of bear is best?", "Sorry I can't answer that question"
    )
    assert len(list(filter(lambda x: x.startswith("prompt_and_output"), result.keys()))) == 1
    assert len(list(filter(lambda x: x.startswith("prompt"), result.keys()))) == 8
    assert len(list(filter(lambda x: x.startswith("output"), result.keys()))) == 7
