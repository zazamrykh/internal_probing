import logging
logger = logging.getLogger(__name__)
from typing import List, Optional, Tuple, Union

import torch
from transformers import GenerationConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate(model,
             tokenizer,
             prompts : Union[Optional[List[torch.long]], List[str]],
             gen_cfg : GenerationConfig = None,
             return_ids = False,
             only_generated = False) -> Tuple[List[str], List[float]]:

    if gen_cfg is None:
        gen_cfg = GenerationConfig(num_beams=1,
                                   do_sample=False,
                                    return_dict_in_generate=True,  # mandatory for scores return
                                    output_scores=True)

    if not isinstance(prompts, list):
        raise ValueError('Prompts suppose to be List instance!')

    if isinstance(prompts[0], str):
        assert tokenizer is not None, 'Input is string but tokenizer is None, you should provide tokenizer or input_ids'
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    elif torch.is_tensor(prompts[0]):
        inputs = {"input_ids": torch.stack(prompts).to(device)}
    else:
        raise ValueError('prompts suppose to be wheather list of ids or list of str')

    gen_out = model.generate(**inputs, generation_config=gen_cfg)

    # logger.debug(f'Output of generatie method: {gen_out}')

    sequences = gen_out.sequences
    scores = gen_out.scores

    if only_generated:
        prompt_lengths = inputs['attention_mask'].sum(dim=1)
        generated_only_sequences = []
        for i in range(sequences.size(0)):
            start = prompt_lengths[i].item()
            generated_only_sequences.append(sequences[i, start:])
        sequences = generated_only_sequences

    if not return_ids:
        generated_texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        return generated_texts, scores
    else:
        return sequences, scores
