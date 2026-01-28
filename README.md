At the current moment framework mostly not implemented. But here I have as for me quite good, relatively short and easy (both gpu and difficulty poin of view) notebooks with semantic entropy and semantic entropy probes demonstration. You can find them at

- notebooks/semantic_entropy.ipynb
- notebooks/semantic_entropy_probes.ipynb

directories. SE notebook shows light example of method, using function from original repository, SEP notebook is larger but complete, It repeats many steps from SEP paper, but for one model and for one dataset.

I have 32 gb RAM and 3060 nvidia 12 gb VRAM. For demo notebooks it is suffisient.

### Internal states probing for prediction uncertainity

Answer accuracy ~ hallucination ~ uncertainty

### Main modules:

1) Scoring

Description: estimators of uncertainity of model answer for particulary prompt or predict other score. Greated score means more hallucination probability / more uncertainty / more diversity in answers / more inaccurate answers.

Examples: semantic entropy, kernel language entropy, inside, reward model, naive entropy, log likelyhood, accuracy, linear probes internal states.

ScorerInterface. Signature:

(ScorerInput) -> (score)

ScorerInput is different for each type of scorer. It inherits ScorerInput base class

SamplerInterface(inherits ScorerInterface). Signature:
(SamplerInput) -> (score). Commonly it invoces model multiple times and define how diffirent generations are.

SamplerInput is prompts and ModelWrapper

LinearProbeScorer(inherits ScorerInterface). Signature:
(ProbeInput) -> (score)
ProbeInput is activations (torch Tensor)

2) Model

Description: model wrapper with prepare_inputs, generate and extract_activations_reforward methods
Main purpose is unification of generation process. For example gpt2 has no apply_chat_template functionality while modern models have.

Other .py files are implementation of base ModelWrapper class

3) Dataset

Module for unified format of datasets. Base class is BaseDataset inside src/dataset/base. Data is storing like list of dicts with fields (optional) written in DatasetSample.


### Run experiments

Many experiments can be run in several stages. Experiment includes stages:

1. Model loading (required)
  - Load base LLM for which correctness will be predicted
  - Wrap it as ModelWrapper
2. Dataset creation or loading. (required)
  - you can directly specify path to enriched_dataset through config or apply enrichers to obtain features from dataset like semantic enropy, greedy answer or activations
3. Method apply (with or without train)
  - Methods could be: linear probe train, prompt embedding probe train
  - For consistency I want to that stage be supportive for sampling based methods or log prob methods like direct semantic entropy calculation (no need train, just sampling) or log prob use for confidence calculation.
4. Metrics calculation

Main regimes of work:
1) Just dataset enrichment.
You can onse run 1-st and 2-nd stages to obtain enriched dataset (dataset for one particular model with greedy answer, semantic entropy sampled answers, se calculater, is_correct field) and then use it for different experiments using specified enriched path.

2) Experiment with some method
Imagine you developed yourth method for predicting correctness (e. g. PEP – prompt embedding probe method) and you want to understand it better. For example, you want to provide different experiments, like choice of batch size, defining required time (or samples_used) for convergence, or understand best layer and best position for predicting is_correct mark. Importang thing for that kind of experiments is returning important information and ability of providing custom flags and parameters for that method (experiment)

3) Main regime
The main point of that repository and experiment runner is
- Provide dataset (e. g. TriviaQA)
- Provide base model (e. g. Mistral 7b)
- Provide method for prediction correctness (e. g. Probes)

Method should return quality (auc metric) measured at test dataset.

### Config

Based on what do I want from experiment runner, I want config to have such sections:
- [model] base llm model parameters
- [dataset] dataset based parameters (name, n_samples, etc)
  - [enricher] could just have enriched_path = <path> at that case just upload dataset from path
  - or enricher section could be list of enricher staged. Based on them factory for enricher steps will be used to add to dataset required fields
- [method] method for prediction of correctness
Contains specific parameters for method. For example, for probes it can contains
  - specific layer or position for probing. Or define_best = True and in such case linear probe will be constructed and trained for every position and layer and then best will be used for metric calculation.
- [metrics] may be useless, but if needed there can be written parameters for metric calculation


License: MIT

## Third-party code

This repository includes third-party code in `external/`:

- OATML/semantic-entropy-probes (MIT License)
- lorenzkuhn/semantic_uncertainty (MIT License)
