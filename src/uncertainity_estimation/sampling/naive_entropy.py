# T = 1, P = 0.9, top_k = 50
import torch
from transformers import GenerationConfig

import logging
logger = logging.getLogger(__name__)

from typing import List, Optional, Union
from uncertainity_estimation.sampling.base import SamplerInterface
from utils.generation import generate
from src.utils.func import scores_to_log_probs


class NaiveEntropySampler(SamplerInterface):
    def __init__(self, n_samples=10, sampling_batch_size=1, generation_config: Optional[GenerationConfig]=None):
        super().__init__(n_samples=n_samples, sampling_batch_size=sampling_batch_size, generation_config=generation_config)


    def generate_samples(self, model, prompts: Union[List[List[torch.long]], List[str]], tokenizer=None):
        output_ids_list, scores_list, log_probs_list = [], [], []

        logger.debug(f'Start generating {self.n_samples} samples for each of {len(prompts)} prompts with batch size {self.sampling_batch_size}')
        prompts_flattened = [prompt for prompt in prompts for _ in range(self.n_samples)]
        logger.debug(f'Total generated sequences will be: {len(prompts_flattened)}')

        for i in range(0, len(prompts_flattened), self.sampling_batch_size):
            logger.debug(f'Passing prompts from index {i} to index {i + self.sampling_batch_size}')
            input_prompts = prompts_flattened[i : min(i + self.sampling_batch_size, len(prompts_flattened))]

            output_ids, scores = generate(model, tokenizer, input_prompts, self.generation_config, return_ids = True)

            log_probs = scores_to_log_probs(scores, output_ids)  # Here we got log_probs – lists with log probs of each generated token
            log_probs_list += log_probs
            output_ids_list += output_ids
            scores_list += scores
        logger.debug(f'First generated sequence is {output_ids_list[0]} with scores {scores_list[0]}')

        # Now group prompts
        scores_list_grouped = []
        for i in range(len(prompts)):
            scores_list_grouped = [scores_list[i : i + self.n_samples]]

        return scores_list_grouped  # list of 'shape' [len(prompts), n_samples, gen_seq_len]


    def compute_consistency(self, generated: Union[List[str], List[List[torch.long]], List[torch.Tensor]]):
        """ generated is scores list from above generate_samples method"""
        return
