"""JEV models (v0) — predict a full-class latent distribution; every question is a marginalization.

Both models emit per-class latent logits z in R^K over the FIXED class universe. A question is a
partition of the K classes into answer groups (choice: each named class + an OUT group of the
omitted classes; binary: {B, not-B}; score: level groups). The answer distribution is

    q_g = sum_{k in group g} softmax(z)_k ,   equivalently   log q = log_softmax_g( LSE_{k in g} z_k )

so OUT is `logsumexp` over the omitted classes (DESIGN.md sec 6). Because groups partition all K
classes, softmax over the grouped logits == the marginalized full-class softmax; named-class odds
are `exp(z_a - z_b)`, independent of the option set (invariance by construction). z does not depend
on the question, so question-isolation is structural.

  AdditiveModel      learned log-linear reference: z_k = log pi_k + sum_f W[f, v_f, k] + b_k
                     (the true log-posterior is exactly this, so it is the fidelity ceiling)
  EncoderQueryModel  transformer encoder over (prior + evidence) tokens; K class-queries
                     cross-attend to it -> z. The intended architecture.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F

NEG_INF = -1e9


def grouped_log_probs(z: torch.Tensor, G: torch.Tensor, g_mask: torch.Tensor) -> torch.Tensor:
    """z:(B,K)  G:(B,Gmax,K) in {0,1}  g_mask:(B,Gmax) bool -> log q:(B,Gmax) (masked groups=-inf).

    Answer logit for group g = logsumexp over its member classes (this is the OUT marginalization);
    then log_softmax over groups. Since the groups partition {0..K-1}, this equals the marginalized
    full-class log-softmax."""
    zc = z.unsqueeze(1).expand(-1, G.size(1), -1)          # (B,Gmax,K)
    masked = torch.where(G > 0, zc, torch.full_like(zc, NEG_INF))
    alog = torch.logsumexp(masked, dim=2)                  # (B,Gmax) grouped logits
    alog = torch.where(g_mask, alog, torch.full_like(alog, NEG_INF))
    return torch.log_softmax(alog, dim=1)


def arm_loss(log_q: torch.Tensor, q_soft: torch.Tensor, g_mask: torch.Tensor,
             arm: str, r_full: torch.Tensor, G: torch.Tensor, gen: torch.Generator | None = None):
    """Loss for one of the three arms. Targets are formed AFTER grouping (DESIGN.md sec 6):
      soft     : cross-entropy to the grouped posterior q_soft = G @ r_full
      sampled  : sample y ~ r_full, label = the group containing y (fresh each step)
      argmax   : label = argmax_g q_soft  (grouped argmax, != argmax latent class for OUT groups)
    """
    if arm == "soft":
        return -(q_soft * log_q.clamp_min(NEG_INF)).masked_fill(~g_mask, 0.0).sum(1).mean()
    if arm == "sampled":
        y = torch.multinomial(r_full, 1).squeeze(1)                     # (B,) global RNG (device-safe)
        g_star = G[torch.arange(G.size(0)), :, y].argmax(1)             # group holding y
    elif arm == "argmax":
        g_star = q_soft.argmax(1)
    else:
        raise ValueError(arm)
    return F.nll_loss(log_q, g_star)


class AdditiveModel(nn.Module):
    """z_k = log(prior_k) + sum_{revealed f} W[f, v_f, k] + b_k.  Exact-capable reference."""
    def __init__(self, F_: int, V: int, K: int):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(F_, V, K))
        self.b = nn.Parameter(torch.zeros(K))
        self.prior_gain = nn.Parameter(torch.ones(()))       # lets it learn to trust the stated prior

    def forward(self, feat, val, mask, prior):               # feat,val,mask:(B,L) ; prior:(B,K)
        w = self.W[feat, val]                                 # (B,L,K)
        z = (w * mask.unsqueeze(-1)).sum(1) + self.b          # (B,K)
        return z + self.prior_gain * torch.log(prior.clamp_min(1e-12))


class EncoderQueryModel(nn.Module):
    """Encoder over prior+evidence tokens; K class-queries cross-attend to it -> z (option-independent)."""
    def __init__(self, F_: int, V: int, K: int, d=128, layers=3, heads=4, ff=256):
        super().__init__()
        self.K = K
        self.class_emb = nn.Embedding(K, d)                  # class identity (prior tokens + queries)
        self.feat_emb = nn.Embedding(F_, d)
        self.val_emb = nn.Embedding(V, d)
        self.type_emb = nn.Embedding(2, d)                   # 0=prior token, 1=evidence token
        self.prior_proj = nn.Linear(1, d)
        enc = nn.TransformerEncoderLayer(d, heads, ff, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.query = nn.Embedding(K, d)                      # one query per latent class
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.head = nn.Linear(d, 1)

    def forward(self, feat, val, mask, prior):
        B, L = feat.shape
        ar = torch.arange(self.K, device=feat.device)
        prior_tok = (self.class_emb(ar)[None].expand(B, -1, -1)
                     + self.prior_proj(prior.unsqueeze(-1))
                     + self.type_emb(torch.zeros(1, dtype=torch.long, device=feat.device)))
        ev_tok = self.feat_emb(feat) + self.val_emb(val) + self.type_emb(
            torch.ones(1, dtype=torch.long, device=feat.device))
        tokens = torch.cat([prior_tok, ev_tok], dim=1)                       # (B, K+L, d)
        pad = torch.cat([torch.zeros(B, self.K, dtype=torch.bool, device=feat.device),
                         ~mask], dim=1)                                       # True = ignore
        H = self.encoder(tokens, src_key_padding_mask=pad)
        q = self.query(ar)[None].expand(B, -1, -1)                           # (B,K,d)
        a, _ = self.cross(q, H, H, key_padding_mask=pad)
        h = self.norm(q + a)
        h = self.norm(h + self.ff(h))
        return self.head(h).squeeze(-1)                                      # (B,K) = z
