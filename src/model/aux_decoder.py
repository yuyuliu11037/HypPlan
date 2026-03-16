from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.prefix_proj = nn.Linear(hidden_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        t_i: torch.Tensor,
        step_token_embeddings: torch.Tensor,
        step_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Prefix t_i as token-0, teacher force with shifted step embeddings.
        prefix = self.prefix_proj(t_i).unsqueeze(1)
        decoder_input = torch.cat([prefix, step_token_embeddings[:, :-1, :]], dim=1)
        seq_len = decoder_input.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=decoder_input.device, dtype=torch.bool),
            diagonal=1,
        )

        hidden = self.decoder(decoder_input, mask=causal_mask)
        logits = self.lm_head(hidden)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            step_token_ids.reshape(-1),
            ignore_index=-100,
        )

