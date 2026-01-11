import logging
logger = logging.getLogger(__name__)
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import GenerationConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_inputs(model, tokenizer, prompts: List[str], model_name=None):
    if model_name is None:
        model_type = model.config.model_type
    else:
        if 'qwen3' in model_name.lower():
            model_type = 'qwen3'
        elif 'gpt2' in model_name.lover():
            model_type = 'gpt2'
        else:
            model_type = 'unknown'
            logger.warning(f'Not implemented get_inputs function for obtaining for that model. Please implement it.')

    logger.debug(f'Defined model name is {model_type}')
    if model_type == 'qwen3':
        messages = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            padding_side='left',
            enable_thinking=False
        ).to(model.device)

        return inputs

    elif model_type == 'gpt2':
        return tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    else:
        messages = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            padding_side='left'
        ).to(model.device)

        return inputs


def truncate_output(outputs, input_ids, tokenizer, return_log_prob=True):
    trunc_seq_list = []

    for i, seq in enumerate(outputs['sequences']):
        first_idx = len(input_ids[i])
        if seq[-1] == tokenizer.eos_token_id or seq[-1] == tokenizer.pad_token_id:
            last_idx = torch.nonzero(seq == tokenizer.eos_token_id)[-1]
        else:
            last_idx = len(seq)
        trunc_seq = seq[first_idx : last_idx + 1]

        trunc_seq_list.append(trunc_seq)

    result_list = [[] for _ in range(len(trunc_seq_list))]
    for step, logits in enumerate(outputs['logits']):
        for i, seq in enumerate(trunc_seq_list):
            if step >= len(seq):
                continue

            if return_log_prob:
                result_list[i].append(F.log_softmax(logits[i], dim=0)[seq[step]])
            else:
                result_list[i].append(logits[i])

    if not return_log_prob:
        return trunc_seq_list, result_list

    result_tensors_list = []
    for trunc_seq, scores in zip(trunc_seq_list, result_list):
        assert len(trunc_seq) == len(scores)
        result_tensors_list.append(torch.tensor(scores))

    return trunc_seq_list, result_tensors_list


def generate(model,
             tokenizer,
             prompts : Union[Optional[List[torch.long]], List[str]],
             gen_cfg : GenerationConfig = None,
             return_ids = False,
             only_generated = False,
             model_name = None,
             return_log_prob = True) -> Tuple[Union[List[str], List[List[torch.long]]], List[List[float]]]:
    if gen_cfg is None:
        gen_cfg = GenerationConfig(num_beams=1,
                                   do_sample=False,
                                    return_dict_in_generate=True,  # mandatory for scores return
                                    output_logits=True)

    if not isinstance(prompts, list):
        raise ValueError('Prompts suppose to be List instance!')

    if isinstance(prompts[0], str):
        assert tokenizer is not None, 'Input is string but tokenizer is None, you should provide tokenizer or input_ids'
        inputs = get_inputs(model, tokenizer, prompts, model_name)
    elif torch.is_tensor(prompts[0]):
        inputs = {"input_ids": torch.stack(prompts).to(device)}
    else:
        raise ValueError('prompts suppose to be wheather list of ids or list of str')

    # logger.debug(f'Inputs in def generate are {inputs} with gen cfg: {gen_cfg}')

    outputs = model.generate(**inputs, generation_config=gen_cfg)

    # logger.debug(f'Output of generate method: {outputs}')

    sequences = outputs.sequences        # (batch, prompt_len + T)     # tuple(len = T) of (batch, vocab)
    batch_size = sequences.size(0)

    if not only_generated:
        # not debuged
        seq_list = [sequences[i] for i in range(batch_size)]
        logits_list = outputs['logits']
        if not return_ids:
            texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
            return texts, logits_list
        else:
            return seq_list, logits_list

    trunc_seq_list, scores_list = truncate_output(outputs, inputs['input_ids'], tokenizer, return_log_prob=return_log_prob)

    if not return_ids:
        texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in trunc_seq_list]
        return texts, scores_list
    else:
        return trunc_seq_list, scores_list

a = 3
