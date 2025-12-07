import torch
device = 'cuda' if torch.cuda.is_available() else ' cpu'
import logging
logger = logging.getLogger(__name__)

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.utils.generation import generate


def test_generate():
    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    texts, scores = generate(
        model,
        tokenizer,
        prompts=["Hello", "London is the capital of"],
        gen_cfg=GenerationConfig(
            max_new_tokens=5,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )
    )

    logger.debug(f'Generated sequence with only_generated = False: {texts}. Result scores: {scores}')

    assert isinstance(texts, list)
    assert len(texts) == 2

    texts, scores = generate(
        model,
        tokenizer,
        prompts=["Hello", "London is the capital of"],
        gen_cfg=GenerationConfig(
            max_new_tokens=5,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        ),
        only_generated = True
    )

    logger.debug(f'Generated sequence with only_generated = False: {texts}. Result scores: {scores}')

    assert isinstance(texts, list)
    assert len(texts) == 2
