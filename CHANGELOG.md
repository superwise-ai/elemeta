## [1.2.1](https://github.com/superwise-ai/elemeta/compare/1.2.0...1.2.1) (2023-12-11)


### <!-- 2. -->:bug: Bug Fixes

* **InjectorSimilarityExtractor:** removed injection ([ed4206f](https://github.com/superwise-ai/elemeta/commit/ed4206fd5a6b83205f63787c3d25c9938a899615))
* **JailbreakSimilarityExtractor:** remove jailbreak ([b45c079](https://github.com/superwise-ai/elemeta/commit/b45c079f4c464500739ed99a7a6295e0689d46b0))
* **JailbreakSimilarityExtractor:** remove jailbreak ([6f3ddc1](https://github.com/superwise-ai/elemeta/commit/6f3ddc1be31a54d50461c98837eb1f1048fc366b))
* **ToxicityExtractor:** toxicity model ([a53e8dd](https://github.com/superwise-ai/elemeta/commit/a53e8ddd0a31632686ae4464e49879bec7c1479b))


### <!-- 4. -->:tractor: Refactor

* pre-commit ([cfd9c49](https://github.com/superwise-ai/elemeta/commit/cfd9c4990a1ed0d9585f75c6309ea2252aa017ba))

## [1.2.0](https://github.com/superwise-ai/elemeta/compare/1.1.2...1.2.0) (2023-12-04)


### <!-- 1. -->:rocket: New Features

* **extractor:** added PII extractor ([ab07e51](https://github.com/superwise-ai/elemeta/commit/ab07e516304806e8355c0831db683d0d55816f14))
* **extractor:** toxicity handle long text ([11d6657](https://github.com/superwise-ai/elemeta/commit/11d6657a216716a4477a895cdba796bf2c3ce40c))


### <!-- 2. -->:bug: Bug Fixes

* **ner identifier:** Changing the name of the extractor to the appropriate name. Using the correct extractors in the test file ([ccc8b05](https://github.com/superwise-ai/elemeta/commit/ccc8b05628f428907d37ddd88c877a90e9fc2b32))
* spacy dependency ([b83094a](https://github.com/superwise-ai/elemeta/commit/b83094a973f5940499a18123b5ae61b01b1e5509))


### <!-- 5. -->:memo: Documentation

* **suites:** add suites to docs ([361834e](https://github.com/superwise-ai/elemeta/commit/361834ecd30e8e5c3c8c9bd1df2a2047ac5edb93))


### <!-- 6. -->:broom: Chore

* **test:** add tests for ToxicityExtractor ([f3b2b33](https://github.com/superwise-ai/elemeta/commit/f3b2b3390112dded2b7b4b2de3e27fdf53ec828b))
* **toxicity_extractor:** extractor fixes to match huggingface output change ([89244b3](https://github.com/superwise-ai/elemeta/commit/89244b37fb983a4f46580179b2552c3cec1e08f0))

## [1.1.2](https://github.com/superwise-ai/elemeta/compare/1.1.1...1.1.2) (2023-10-04)


### <!-- 2. -->:bug: Bug Fixes

* **common:** move the common folder to be inside the package ([02013ab](https://github.com/superwise-ai/elemeta/commit/02013ab37afa2dbfa25758a3085932494aaf6cd3))
* **suites:** add __init__.py to the suites folder ([7083870](https://github.com/superwise-ai/elemeta/commit/7083870ed0832a10a18ee6d8f70ff403447c3594))


### <!-- 4. -->:tractor: Refactor

* **PairMetafeatureExtractorsRunnerResult:** will have the name of the extractor near the value ([44bdfc4](https://github.com/superwise-ai/elemeta/commit/44bdfc4cc0db3a2066a96ca786786786934a44a0))


### <!-- 6. -->:broom: Chore

* **tests:** add more tests ([6e16828](https://github.com/superwise-ai/elemeta/commit/6e168284a9671c3f6bed7501bb72ed0f0aa02c86))

## [1.1.1](https://github.com/superwise-ai/elemeta/compare/1.1.0...1.1.1) (2023-10-03)


### <!-- 2. -->:bug: Bug Fixes

* **docs:** fix docs for the changes for the new version and rename the input parameter in few extractors ([33a3bc2](https://github.com/superwise-ai/elemeta/commit/33a3bc2e9eb4d48f472e94e7763a1881f2aab377))

## [1.1.0](https://github.com/superwise-ai/elemeta/compare/1.0.7...1.1.0) (2023-10-02)


### <!-- 1. -->:rocket: New Features

* **AbstractTextPairMetafeatureExtractor:** add AbstractTextPairMetafeatureExtractor ([647724c](https://github.com/superwise-ai/elemeta/commit/647724ca90f0e2e56aff76649c5b7de9919826d5))
* **metadata_extractors:** add embedding extraction funtion and many embedding and text similarity metrics ([a2343ff](https://github.com/superwise-ai/elemeta/commit/a2343ff1baf7a085d5052cf1a6e6f05ec2ef0bc6))
* **PII_identifier:** add PII identifying extractor ([1546ad9](https://github.com/superwise-ai/elemeta/commit/1546ad95209bf3b5a8d8cafc6761a2367d8e7183))
* **poetry:** add torch and transformers to Poetry ([8bc1b22](https://github.com/superwise-ai/elemeta/commit/8bc1b2271feb4e54ae6f073d921724369f2a1cb0))
* **suites:** add CommonLLMSuite and some fixes ([013a659](https://github.com/superwise-ai/elemeta/commit/013a6595021435043bbbbe711b23c53e75df70c1))
* **test:** add test for PairMetafeatureExtractorsRunner ([d42fa6c](https://github.com/superwise-ai/elemeta/commit/d42fa6cc19a8a559ded1e47583c41e82dd8b35fc))
* **toxicity_measure:** Addition of ToxicityExtractor and Test ([eaa41ee](https://github.com/superwise-ai/elemeta/commit/eaa41ee00e1b615f9fcc1cb973998f6f4c8f5b80))


### <!-- 2. -->:bug: Bug Fixes

* **extractor_runner:** changes to allow run_on_dataframe to work with ToxicitiyExtractor ([641de63](https://github.com/superwise-ai/elemeta/commit/641de631b5c7946a47dc7094500575249e5e655f))


### <!-- 5. -->:memo: Documentation

* **docs:** add docs to the new extractors ([5a0019b](https://github.com/superwise-ai/elemeta/commit/5a0019bf5dbf697b04b47b2d736fe2a763a20963))


### <!-- 6. -->:broom: Chore

* **test:** update tests and fix few bugs ([12e9e91](https://github.com/superwise-ai/elemeta/commit/12e9e9119caef2d546afcb03a98424b16f160e10))

## [1.0.7](https://github.com/superwise-ai/elemeta/compare/1.0.6...1.0.7) (2023-09-12)


### <!-- 2. -->:bug: Bug Fixes

* **build:** fix build.sh script ([778dd77](https://github.com/superwise-ai/elemeta/commit/778dd7714d4581212ca091001b01623ec474fa7e))
* **docs:** fix typo in CONTRIBUTING.md ([22c2e1e](https://github.com/superwise-ai/elemeta/commit/22c2e1e2568b23714924ad56b69f52f6d67be0d0))


### <!-- 6. -->:broom: Chore

* **formatting:** Apply pre-commit formmating ([3f3c1a1](https://github.com/superwise-ai/elemeta/commit/3f3c1a1c0746f14e0632b3855547906fcfa5c363))
* **formatting:** Use pre-commit for lint & format ([f2a4261](https://github.com/superwise-ai/elemeta/commit/f2a42618440254844094879719d09e48334e916f))

## [1.0.6](https://github.com/superwise-ai/elemeta/compare/1.0.5...1.0.6) (2023-05-14)


### <!-- 6. -->:broom: Chore

* **ci:** Fix CI triggers for PRs ([a93d429](https://github.com/superwise-ai/elemeta/commit/a93d4291981999bf9321b1f3f53ef177be13ba65))
* **contributing:** Adding CONTRIBUTING guidelines ([9fcc582](https://github.com/superwise-ai/elemeta/commit/9fcc5823f4937bc3e1dc0a9ffe96dd98012386bc))
* **contributing:** Update the commit format ([3191f9c](https://github.com/superwise-ai/elemeta/commit/3191f9c88cea9f81dcaf567138d791e448211424))
* **elemeta:** Changed DetectLangauge to DetectLanguage in the docs and code ([31e09a0](https://github.com/superwise-ai/elemeta/commit/31e09a08ecc5fe7543d285cd3307f617ee9a156a))


### <!-- 5. -->:memo: Documentation

* **formatting:** Fix AbstractTextMetafeatureExtractor indentation ([0fbf0f1](https://github.com/superwise-ai/elemeta/commit/0fbf0f1461e00ae27712c0f8ed4dfb66996e92d7))


### <!-- 2. -->:bug: Bug Fixes

* **OutOfVocabularyCount:** change the tokenizer to ignore punctuation ([2b0fecb](https://github.com/superwise-ai/elemeta/commit/2b0fecb4e953f56ce07f5c384429e766bd9cc082))

## [1.0.5](https://github.com/superwise-ai/elemeta/compare/1.0.4...1.0.5) (2023-04-25)


### Bug Fixes

* **extractor_runner:** adding compute_intensive flag to constructor and add imporve tests ([4e38299](https://github.com/superwise-ai/elemeta/commit/4e38299400f64e1842e3c5ba0b2b26b174bae047))

## [1.0.4](https://github.com/superwise-ai/elemeta/compare/1.0.3...1.0.4) (2023-04-24)


### Bug Fixes

* **extractor_runner:** DetectLangauge extractor to the list of extractors ([67af4cb](https://github.com/superwise-ai/elemeta/commit/67af4cb6bb195f07f5ef3724186e59cc56b715d4))

## [1.0.3](https://github.com/superwise-ai/elemeta/compare/1.0.2...1.0.3) (2023-04-23)


### Bug Fixes

* **elemeta:** aline terminology for metafeature extractors ([3241614](https://github.com/superwise-ai/elemeta/commit/3241614289e831f037c41f7760ba7b4b3587f894))

## [1.0.2](https://github.com/superwise-ai/elemeta/compare/1.0.1...1.0.2) (2023-04-23)


### Bug Fixes

* **dataset:** compress datasets files and add new tweets_likes dataset ([2e2f54b](https://github.com/superwise-ai/elemeta/commit/2e2f54bd7b013c90ec0ea3d40a95833489f93a0f))

## [1.0.1](https://github.com/superwise-ai/elemeta/compare/1.0.0...1.0.1) (2023-04-04)


### Bug Fixes

* **security:** update minimal python version to 3.8 ([0b812f6](https://github.com/superwise-ai/elemeta/commit/0b812f6bd17281305315f7f607685ee7e906dcda))

# 1.0.0 (2023-04-03)


### Features

* **initial release:** initial release ([e38d3ec](https://github.com/superwise-ai/elemeta/commit/e38d3eccd2c9cd0e50b7ee2a8ea558ed4a392a02))
