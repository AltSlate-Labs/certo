"""Generic decision model: encode the state once, encode each OPTION from its text description, and
score every option independently against the state. Options are runtime -- any number, any wording,
never-seen ones included -- because an option's logit comes from its own description + the shared
state, not from a fixed class identity or from the other options.

  state text ──ModernBERT──▶ H_s
  option_m text ──ModernBERT──▶ pooled o_m ──cross-attend to H_s──▶ context c_m ──MLP──▶ logit_m
  softmax over the presented options.

Because each option queries the state on its own (queries don't attend to each other), the per-option
logit is independent of which other options appear ⇒ odds-invariance / option-order-invariance by
construction (the property Laya lacks). Encoder shared between state and options; both fine-tuned.
"""
from __future__ import annotations
import torch, torch.nn as nn
from transformers import AutoModel

BACKBONE = "answerdotai/ModernBERT-base"
NEG = -1e9


class GenericDecisionModel(nn.Module):
    def __init__(self, backbone: str = BACKBONE, heads: int = 8):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone)
        d = self.bert.config.hidden_size
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.score = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def _enc(self, ids, mask):
        return self.bert(input_ids=ids, attention_mask=mask).last_hidden_state

    @staticmethod
    def _pool(H, mask):
        m = mask.unsqueeze(-1).float()
        return (H * m).sum(1) / m.sum(1).clamp_min(1.0)

    def forward(self, s_ids, s_mask, o_ids, o_mask, opt_valid):
        # s_ids (B,Ls); o_ids (B,M,Lo); opt_valid (B,M) bool
        B, M, Lo = o_ids.shape
        H_s = self._enc(s_ids, s_mask)                                   # (B,Ls,d)
        H_o = self._enc(o_ids.reshape(B * M, Lo), o_mask.reshape(B * M, Lo))
        o_vec = self._pool(H_o, o_mask.reshape(B * M, Lo)).reshape(B, M, -1)   # (B,M,d)
        ctx, _ = self.cross(o_vec, H_s, H_s, key_padding_mask=(s_mask == 0))    # each option reads the state
        h = self.norm(o_vec + ctx)
        logit = self.score(torch.cat([o_vec, h], -1)).squeeze(-1)        # (B,M)
        return logit.masked_fill(~opt_valid, NEG)

    def param_groups(self, bert_lr=2e-5, head_lr=1e-3):
        bert = set(self.bert.parameters())
        new = [p for p in self.parameters() if p not in bert]
        return [{"params": list(bert), "lr": bert_lr}, {"params": new, "lr": head_lr}]
