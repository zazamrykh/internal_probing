### Internal states probing for prediction uncertainity

Answer accuracy ~ hallucination ~ uncertainty

### Requirements
torch of
### Main modules:

1) Uncertainity estimation (ue)

Description: estimators of uncertainity of model answer for particulary prompt.

Examples: semantic entropy, kernel language entropy, inside, reward model, naive entropy, log likelyhood, accuracy, linear probes internal states.

ScorerInterface. Signature:

(model, prompts, answer, activation) -> (score)

SamplerInterface(inherits ScorerInterface). Signature:
(model, prompts, answer=None, activation=None) -> (score)

ProbeInterface(inherits ScorerInterface). Signature:
(model=None, prompts=None, answer=None, activation=not None) -> (score)

2) Activation extractor

Description: extracts activations of model from specific layer / token position / activation part for any model and prompt + answer

ActivationExtractorInterface. Signature:

(model, prompts + answers concat) -> (activation tensor)

It can be implemented with standart HuggingFace transformers lib with output_hidden_states=True. It is simple but consumes more memory (it returns hidden states for each layer) and it is impossible to access concrete mlp or attention activation. Also it can be implemented with TransformerLens library, it is more convinient because of flexibility. But in practice in real time scenario pytorch hooks is more preferred because is fast and do not reqiures additional packages.

3) Dataset

Torch dataset

Description:
- Datasets with common prompts such as BioASQ, TriviaQA, NQ Open, SQuAD. Module suppose to transform datasets to unified formats.
- Datasets with stored extracted activation and assosiated score with meta info about model, layer, token, activation_type, which score used.

4) Metrics

Description: metrics should estimates how well predicted score is correlating with accuracy label. It suppose to infer some method of uncertainty prediction, mark 1 if answer is not confidence and 0 if confident. Than we calculate how many labels coincident with GT – whether model answer is correct. So module gets dataset, model, method UE. It infer model, estimates uncertainty, calculates metrics (auc / prec / rec).

EvalInterface. Signature

(dataset, model, method : ScorerInterface) -> (metrics)

### Tests
--tb=long flag allows traceback all info
You can run tests as follows:
