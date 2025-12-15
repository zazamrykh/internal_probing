import torch
device = 'cuda' if torch.cuda.is_available() else ' cpu'
import logging
logger = logging.getLogger(__name__)

from src.utils.config import Config
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.uncertainity_estimation.sampling.naive_entropy import NaiveEntropySampler

def test_naive_entropy_sampler():
    gen_cfg = GenerationConfig(
        max_new_tokens=32,
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
        top_k=50,
        num_beams=1,
        output_logits=True,
        return_dict_in_generate=True,
    )

    sampler = NaiveEntropySampler(n_samples=5, sampling_batch_size=1, generation_config=gen_cfg)

    test_config_path = 'configs/test_config.ini'
    try:
        config = Config(test_config_path)
    except:
        config = None
        logger.info(f'Cannot load config from {test_config_path}')

    model_name = config.model['model_name'] if config else "sshleifer/tiny-gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    prompts = ['When Pushkin was born?', 'When Saint-Petersburg was found?', 'What is root of equation x^2 -3x + 4 = 0?']

    scores = sampler.generate_samples(model, prompts, tokenizer)
    # TODO: make normal assertion
    logger.debug(f'Obtained scores length by generate_samples method: {len(scores)}')
    # logger.debug(f'Shape of each scores tensor: {[score_tensor.shape for score_tensor in scores]}')
