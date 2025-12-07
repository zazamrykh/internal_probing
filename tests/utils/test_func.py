import torch

from src.utils.func import scores_to_log_probs

def test_scores_to_log_probs_toy():
    batch_size = 2
    vocab_size = 4
    num_steps = 3

    # Toy scores: 3 steps, each (2, 4)
    scores = tuple(
        torch.randn(batch_size, vocab_size)
        for _ in range(num_steps)
    )

    # toy generated tokens
    output_ids = torch.tensor([
        [0, 1, 2],
        [3, 2, 1],
    ])

    log_probs = scores_to_log_probs(scores, output_ids)

    assert isinstance(log_probs, torch.Tensor)
    assert log_probs.shape == (batch_size, num_steps)
