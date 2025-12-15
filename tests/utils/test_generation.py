import torch

from src.utils.config import Config
device = 'cuda' if torch.cuda.is_available() else ' cpu'
import logging
logger = logging.getLogger(__name__)

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.utils.generation import generate


def test_generate():
    test_config_path = 'configs/test_config.ini'
    try:
        config = Config(test_config_path)
    except:
        config = None
        logger.info(f'Cannot load config from {test_config_path}')

    model_name = config.model['model_name'] if config else "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    gen_cfg = GenerationConfig(num_beams=1,
                            do_sample=False,
                            return_dict_in_generate=True,
                            # output_scores=True,
                            output_logits=True)

    texts, scores = generate(
        model,
        tokenizer,
        prompts=["Hello", "London is the capital of"],
        gen_cfg=gen_cfg
    )

    logger.debug(f'Generated sequence with only_generated = False: {texts}. Result scores: {scores}')

    assert isinstance(texts, list)
    assert len(texts) == 2

    texts, scores = generate(
        model,
        tokenizer,
        prompts=["Hello", "London is the capital of"],
        gen_cfg=gen_cfg,
        only_generated = True,
    )

    logger.debug(f'Generated sequence with only_generated = True: {texts}. Result scores: {scores}')

    assert isinstance(texts, list)
    assert len(texts) == 2
