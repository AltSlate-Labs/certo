"""Stage B model: ModernBERT-base encoder + the v0 query reader / typed heads / fixed-class OUT.

The encoder reads the STATE TEXT only (prior + evidence in language); K learned class-queries
cross-attend to its token states to produce per-class logits z. z is question-independent, so
question-isolation and named-class odds-invariance carry over exactly as in the symbolic model
(the question is applied afterwards as a class partition; see model.grouped_log_probs). The prior
is read from the text, so this tests whether a pretrained language encoder can recover the known
probabilities through wording.
"""
from __future__ import annotations
import torch, torch.nn as nn
from transformers import AutoModel

BACKBONE = "answerdotai/ModernBERT-base"


class ModernBERTQueryModel(nn.Module):
    def __init__(self, K: int, backbone: str = BACKBONE, heads: int = 8):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone)
        d = self.bert.config.hidden_size
        self.K = K
        self.query = nn.Embedding(K, d)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.head = nn.Linear(d, 1)

    def forward(self, input_ids, attention_mask):
        H = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        B = input_ids.size(0)
        q = self.query(torch.arange(self.K, device=input_ids.device))[None].expand(B, -1, -1)
        pad = attention_mask == 0
        a, _ = self.cross(q, H, H, key_padding_mask=pad)
        h = self.norm(q + a)
        h = self.norm(h + self.ff(h))
        return self.head(h).squeeze(-1)                 # (B, K) latent-class logits z

    def param_groups(self, bert_lr=2e-5, head_lr=1e-3):
        """Fine-tune the encoder slowly; train the new reader/heads faster."""
        bert = set(self.bert.parameters())
        new = [p for p in self.parameters() if p not in bert]
        return [{"params": list(bert), "lr": bert_lr}, {"params": new, "lr": head_lr}]
