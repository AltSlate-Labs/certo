"""Worked examples: run the trained decision model on concrete cases and look at the probabilities.

    python examples.py <checkpoint_dir> [device]
"""
import sys, numpy as np, torch
import genworld as G
import train_generic as T          # render_state / render_option / NAMES_TEST
from infer import DecisionModel

ckpt = sys.argv[1] if len(sys.argv) > 1 else "ckpt_generic_large"
dev = sys.argv[2] if len(sys.argv) > 2 else ("cuda" if torch.cuda.is_available() else "cpu")
dm = DecisionModel.load(ckpt, device=dev)
cfg = G.GenConfig(); rng = np.random.default_rng(7)


def make(e, template=3):
    order = list(np.nonzero(e.reveal)[0]); rng.shuffle(order)
    names = list(rng.choice(T.NAMES_TEST, size=e.protos.shape[0], replace=False))
    state = T.render_state(cfg, e, template, order)
    opts = [{"id": names[j], "description": T.render_option(cfg, e.protos[j], names[j], template)}
            for j in range(len(names))]
    truth = {names[j]: round(float(e.r[j]), 3) for j in range(len(names))}
    return state, opts, truth


def show(title, state, opts, truth=None, abstain=0.6):
    r = dm.decide(state, opts, abstain_below=abstain)
    print(f"\n### {title}")
    print("state   :", state[:150] + ("…" if len(state) > 150 else ""))
    print("predicted:", r["probs"], "| top", r["top"], "| abstain", r.get("abstain"))
    if truth:
        print("exact    :", truth)


# 1. clear evidence, with ground truth (unseen options)
while True:
    e = G.sample_example(cfg, rng)
    if G.dist_entropy(e.r) / np.log(len(e.r)) < 0.2:   # confident case
        break
st, op, tr = make(e); show("clear evidence — should be confident & calibrated", st, op, tr)

# 2. ambiguous evidence -> genuine spread
while True:
    e = G.sample_example(cfg, rng)
    if G.dist_entropy(e.r) / np.log(len(e.r)) > 0.7:   # uncertain case
        break
st, op, tr = make(e); show("ambiguous evidence — should be a spread, not false confidence", st, op, tr)

# 3. option-order invariance (same case, options reversed)
st, op, tr = make(G.sample_example(cfg, rng))
r1 = dm.decide(st, op)["probs"]; r2 = dm.decide(st, list(reversed(op)))["probs"]
print("\n### option-order invariance — per-option probs should be identical")
print("original:", r1)
print("reversed:", {k: r2[k] for k in r1})

# 4. natural language (OUT OF DISTRIBUTION) — honest transfer probe
show("natural language (out of training distribution) — expect degradation",
     "Customer: I was charged twice for the same order and I'd like a refund.",
     [{"id": "billing", "description": "billing and payments: duplicate charges, refunds, invoices"},
      {"id": "tech", "description": "the app or website is broken or crashing"},
      {"id": "account", "description": "login and account access problems"}], abstain=0.5)
