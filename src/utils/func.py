from typing import List
import torch
import torch.nn.functional as F

def scores_to_log_probs(
    scores: List[List],       # List[(T_i, vocab)]
    output_ids: List[torch.Tensor],   # List[(T_i,)]
) -> List[torch.Tensor]:
    ### DEPRECATED
    assert len(scores) == len(output_ids)

    out: List[torch.Tensor] = []

    for s, ids in zip(scores, output_ids):
        log_probs = F.log_softmax(s, dim=-1)            # (T_i, vocab)
        step_log_probs = log_probs[
            torch.arange(ids.size(0), device=ids.device),
            ids,
        ]                                               # (T_i,)

        out.append(step_log_probs)

    return out


def calculate_naive_entropy(log_probs):
    pass
