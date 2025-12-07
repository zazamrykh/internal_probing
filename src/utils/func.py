from typing import List, Tuple
import torch
import torch.nn.functional as F

def scores_to_log_probs(
    scores: Tuple[torch.Tensor, ...],
    output_ids: torch.Tensor,
    beam_size = 1,
) -> List[torch.Tensor]:
    assert beam_size == 1, 'Function only works for beam_size = 1'

    log_probs_per_step = [F.log_softmax(s, dim=-1) for s in scores]
    num_steps = len(log_probs_per_step)
    batch_size = output_ids.size(0)

    if output_ids.size(1) < num_steps:
        raise ValueError("output_ids must have at least num_steps columns")

    gen_tokens = output_ids[:, :num_steps]  # (batch, num_steps)

    out = []
    for t in range(num_steps):
        step_log_probs = log_probs_per_step[t]        # (batch, vocab)
        step_tokens = gen_tokens[:, t]                # (batch,)
        out.append(step_log_probs[
            torch.arange(batch_size, device=step_tokens.device),
            step_tokens
        ])

    return torch.stack(out, dim=1)

def calculate_naive_entropy(log_probs):
    pass
