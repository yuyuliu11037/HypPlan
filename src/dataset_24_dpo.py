"""Dataset for DPO preference training on Game of 24 planning vectors.

Each item yields the pieces needed for one preference pair:
- ctx_ids: tokenized context (ends at the step boundary where z is injected)
- pos_ids: positive next-step tokens
- neg_ids: negative next-step tokens
- log_pi_ref_pos/neg: precomputed reference log-probs

Batching is handled by collate_fn with separate padding for each field.
"""
from __future__ import annotations

import json

import torch
from torch.utils.data import Dataset


class Game24DPODataset(Dataset):
    def __init__(self, tokenizer, pairs_path: str, refs_path: str,
                 max_ctx_len: int = 256, max_tail_len: int = 48):
        self.tokenizer = tokenizer
        self.max_ctx_len = max_ctx_len
        self.max_tail_len = max_tail_len

        with open(pairs_path) as f:
            self.pairs = [json.loads(line) for line in f]

        refs = torch.load(refs_path)
        self.log_pi_ref_pos = refs["log_pi_ref_pos"]
        self.log_pi_ref_neg = refs["log_pi_ref_neg"]
        assert len(self.pairs) == self.log_pi_ref_pos.size(0), (
            f"Mismatch: {len(self.pairs)} pairs vs {self.log_pi_ref_pos.size(0)} refs"
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        ctx_ids = self.tokenizer.encode(p["ctx_text"], add_special_tokens=False)
        pos_ids = self.tokenizer.encode(p["pos_tail"], add_special_tokens=False)
        neg_ids = self.tokenizer.encode(p["neg_tail"], add_special_tokens=False)

        if len(ctx_ids) > self.max_ctx_len:
            ctx_ids = ctx_ids[-self.max_ctx_len:]  # keep tail of ctx
        pos_ids = pos_ids[:self.max_tail_len]
        neg_ids = neg_ids[:self.max_tail_len]

        return {
            "ctx_ids": torch.tensor(ctx_ids, dtype=torch.long),
            "pos_ids": torch.tensor(pos_ids, dtype=torch.long),
            "neg_ids": torch.tensor(neg_ids, dtype=torch.long),
            "log_pi_ref_pos": float(self.log_pi_ref_pos[idx].item()),
            "log_pi_ref_neg": float(self.log_pi_ref_neg[idx].item()),
        }


def collate_fn(batch):
    """Trivial collate for batch_size=1: just return the single item unwrapped (as tensors)."""
    assert len(batch) == 1, "DPO training uses batch_size=1 with grad_accum"
    return batch[0]
