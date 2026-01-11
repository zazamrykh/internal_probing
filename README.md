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

### Tests
--tb=long flag allows traceback all info
Some tests related to models would require it. So ypu can place loaclly model to models directory otherwise it will be downloaded.

You should run tests from tests subfolder.
