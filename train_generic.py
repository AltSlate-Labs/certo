"""Stage C / step 1 training + eval: a GENERIC decision model on the runtime-option world.

Renders each example to (state text, M option-description texts), trains the generic model with soft
targets (+ sampled/argmax arms), and grades posterior fidelity (KL/TV) plus the capabilities that
define "generic":
  - unseen options    : test on held-out option PROTOTYPES *and* held-out option NAME vocabulary.
  - more options      : test with more options than ever seen in training.
  - order-invariance  : per-option logits unchanged under permuting the option order.

  python train_generic.py smoke
  python train_generic.py run --arm soft --device cuda --ntr 20000 --epochs 12
"""
from __future__ import annotations
import sys, os, json, argparse, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import genworld as G
from realize import ATTR_NAMES, VALUE_WORDS
from model_generic import GenericDecisionModel, BACKBONE
from train_text import _device

NAMES = ["Zeta", "Nova", "Pike", "Wren", "Cove", "Flint", "Marsh", "Vale", "Reed", "Sage",
         "Birch", "Cliff", "Dune", "Fen", "Glen", "Heath", "Isle", "Kelp", "Loam", "Moor",
         "Peak", "Ridge", "Shoal", "Thaw"]
NAMES_TRAIN, NAMES_TEST = NAMES[:12], NAMES[12:]      # 12 each; test pool >= max option count (10)
TRAIN_TPL, TEST_TPL = (0, 1, 2), (3,)


def render_state(cfg, e, t, order):
    if len(order) == 0:
        return "No evidence has been observed yet."
    items = [(ATTR_NAMES[f], VALUE_WORDS[e.values[f]]) for f in order]
    if t == 0: return "Observed evidence: " + "; ".join(f"{a}={v}" for a, v in items) + "."
    if t == 1: return " ".join(f"Observed {a}: {v}." for a, v in items)
    if t == 2: return " ".join(f"The {a} reading was {v}." for a, v in items)
    return "We measured " + ", ".join(f"{a} as {v}" for a, v in items) + "."


def render_option(cfg, proto, name, t):
    items = [(ATTR_NAMES[f], VALUE_WORDS[proto[f]]) for f in range(cfg.F)]
    body = ", ".join(f"{a} {v}" for a, v in items)
    if t == 0: return f"Option {name}: typically {body}."
    if t == 1: return f"{name} usually shows {body}."
    if t == 2: return f"Profile of {name}: {body}."
    return f"{name} — characteristic readings: {body}."


def build(cfg, n, seed, tpl, names, tok, max_s, max_o, Mmax, M_range=None):
    rng = np.random.default_rng(seed)
    s_txt = []; o_txt = []; r = np.zeros((n, Mmax), np.float32)
    valid = np.zeros((n, Mmax), bool); y = np.zeros(n, np.int64)
    for i in range(n):
        mr = M_range or cfg.M_range
        M = int(rng.integers(mr[0], mr[1] + 1))
        e = G.sample_example(cfg, rng, M=M)
        order = list(np.nonzero(e.reveal)[0]); rng.shuffle(order)
        t = int(rng.choice(tpl)); nm = list(rng.choice(names, size=M, replace=False))
        s_txt.append(render_state(cfg, e, t, order))
        for o in range(M):
            o_txt.append(render_option(cfg, e.protos[o], nm[o], t))
        for _ in range(Mmax - M):
            o_txt.append("Option none.")
        r[i, :M] = e.r; valid[i, :M] = True; y[i] = e.y
    se = tok(s_txt, padding="max_length", truncation=True, max_length=max_s, return_tensors="pt")
    oe = tok(o_txt, padding="max_length", truncation=True, max_length=max_o, return_tensors="pt")
    return dict(s_ids=se["input_ids"], s_mask=se["attention_mask"],
                o_ids=oe["input_ids"].reshape(n, Mmax, max_o), o_mask=oe["attention_mask"].reshape(n, Mmax, max_o),
                valid=torch.from_numpy(valid), r=torch.from_numpy(r), y=torch.from_numpy(y))


def _fwd(model, d, b, dev):
    return model(d["s_ids"][b].to(dev), d["s_mask"][b].to(dev),
                 d["o_ids"][b].to(dev), d["o_mask"][b].to(dev), d["valid"][b].to(dev))


def _unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def save_checkpoint(model, tokenizer, path, meta):
    os.makedirs(path, exist_ok=True)
    torch.save(_unwrap(model).state_dict(), os.path.join(path, "model.pt"))
    tokenizer.save_pretrained(path)
    json.dump(meta, open(os.path.join(path, "certo_config.json"), "w"), indent=2)
    print(f"saved checkpoint -> {path}", flush=True)


def train(model, d, arm, epochs, bs, dev, bert_lr, head_lr, seed=0, warmup=0.06):
    opt = torch.optim.AdamW(_unwrap(model).param_groups(bert_lr, head_lr))
    gen = torch.Generator().manual_seed(seed); n = d["y"].size(0); model.train()
    sched = get_linear_schedule_with_warmup(opt, int(warmup * epochs * ((n + bs - 1)//bs)), epochs*((n+bs-1)//bs))
    for ep in range(epochs):
        tot = 0.0; nb = 0
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, bs):
            b = perm[s:s+bs]
            logit = _fwd(model, d, b, dev)
            logq = torch.log_softmax(logit, 1)
            r = d["r"][b].to(dev)
            if arm == "soft":     loss = -(r * logq).sum(1).mean()
            elif arm == "sampled": loss = F.nll_loss(logq, torch.multinomial(r, 1).squeeze(1))
            elif arm == "argmax":  loss = F.nll_loss(logq, r.argmax(1))
            else: raise ValueError(arm)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item(); nb += 1
        print(f"  epoch {ep+1}/{epochs}  loss={tot/max(nb,1):.4f}  lr={sched.get_last_lr()[0]:.2e}", flush=True)
    return model


@torch.no_grad()
def evaluate(model, d, dev, bs=128, nbins=15):
    model.eval(); L = []
    for s in range(0, d["y"].size(0), bs):
        L.append(_fwd(model, d, torch.arange(s, min(s+bs, d["y"].size(0))), dev).cpu())
    logit = torch.cat(L); valid = d["valid"]; r = d["r"]; y = d["y"]
    p = torch.softmax(logit, 1)
    rp = r.clamp_min(1e-12)
    kl = torch.where(valid, r * (rp.log() - p.clamp_min(1e-12).log()), torch.zeros_like(r)).sum(1).mean().item()
    tv = 0.5 * (r - p).abs().masked_fill(~valid, 0).sum(1).mean().item()
    acc = (p.argmax(1) == y).float().mean().item()
    onehot = torch.zeros_like(p).scatter_(1, y[:, None], 1.0)
    conf = p[valid]; hit = onehot[valid]; bins = torch.clamp((conf*nbins).long(), 0, nbins-1)
    ece = sum((bins == b).float().mean().item() * abs(conf[bins==b].mean()-hit[bins==b].mean()).item()
              for b in range(nbins) if (bins == b).any())
    return dict(KL=kl, TV=tv, acc=acc, ECE=ece)


@torch.no_grad()
def order_invariance(model, d, dev, Mval=4, n=256):
    model.eval()
    sel = torch.nonzero(d["valid"].sum(1) == Mval).squeeze(1)[:n]
    if len(sel) < 8: return float("nan")
    sub = {"s_ids": d["s_ids"][sel], "s_mask": d["s_mask"][sel],
           "o_ids": d["o_ids"][sel][:, :Mval], "o_mask": d["o_mask"][sel][:, :Mval],
           "valid": d["valid"][sel][:, :Mval]}
    idx = torch.arange(len(sel))
    base = _fwd(model, sub, idx, dev).cpu()
    perm = torch.randperm(Mval)
    sub2 = dict(sub); sub2["o_ids"] = sub["o_ids"][:, perm]; sub2["o_mask"] = sub["o_mask"][:, perm]
    permd = _fwd(model, sub2, idx, dev).cpu()
    return (permd - base[:, perm]).abs().max().item()


def run(a):
    dev = _device(a.device); cfg = G.GenConfig()
    tok = AutoTokenizer.from_pretrained(a.backbone)
    ms, mo = 64, 48
    print(f"Stage C step1 (generic)  arm={a.arm} dev={dev} Ntr={a.ntr} epochs={a.epochs}\n", flush=True)
    tr = build(cfg, a.ntr, a.seed, TRAIN_TPL, NAMES_TRAIN, tok, ms, mo, 6)
    te = build(cfg, a.nte, a.seed+1, TEST_TPL, NAMES_TEST, tok, ms, mo, 6)          # unseen states+templates+names
    te_more = build(cfg, a.nte, a.seed+2, TEST_TPL, NAMES_TEST, tok, ms, mo, 10, M_range=(8, 10))
    torch.manual_seed(a.seed); model = GenericDecisionModel(a.backbone).to(dev)
    if dev == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"DataParallel across {torch.cuda.device_count()} GPUs", flush=True)
    train(model, tr, a.arm, a.epochs, a.bs, dev, a.bert_lr, a.head_lr, a.seed)
    rm = evaluate(model, te, dev); rmo = evaluate(model, te_more, dev); oi = order_invariance(model, te, dev)
    print(f"\n{'eval':<34}{'KL(r|p)':>9}{'TV':>8}{'acc':>7}{'ECE':>8}")
    print("-"*66)
    print(f"{'unseen states+templates+names':<34}{rm['KL']:>9.4f}{rm['TV']:>8.4f}{rm['acc']:>7.3f}{rm['ECE']:>8.4f}")
    print(f"{'+ more options (8-10, seen 3-6)':<34}{rmo['KL']:>9.4f}{rmo['TV']:>8.4f}{rmo['acc']:>7.3f}{rmo['ECE']:>8.4f}")
    print(f"\n[{'PASS' if oi < 1e-3 else 'FAIL'}] option-order invariance  max|Δlogit|={oi:.2e}")
    print("success = low KL on BOTH eval rows (generalizes to unseen options AND larger option sets).")
    if a.save:
        save_checkpoint(model, tok, a.save, {
            "kind": "generic", "backbone": a.backbone, "heads": 8, "arm": a.arm,
            "trained_on": "genworld", "ntr": a.ntr, "epochs": a.epochs,
            "eval": {"unseen": rm, "more_options": rmo, "order_inv": oi}})


def smoke():
    dev = _device("auto"); print("smoke dev:", dev, flush=True); cfg = G.GenConfig()
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    tr = build(cfg, 64, 0, TRAIN_TPL, NAMES_TRAIN, tok, 64, 48, 6)
    te = build(cfg, 64, 1, TEST_TPL, NAMES_TEST, tok, 64, 48, 6)
    m = GenericDecisionModel().to(dev); train(m, tr, "soft", 1, 16, dev, 2e-5, 1e-3)
    print("eval:", {k: round(v, 3) for k, v in evaluate(m, te, dev).items()})
    print("order-inv:", f"{order_invariance(m, te, dev):.2e}"); print("SMOKE OK")


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=["soft", "sampled", "argmax"], default="soft")
    r.add_argument("--device", default="auto"); r.add_argument("--backbone", default=BACKBONE)
    r.add_argument("--ntr", type=int, default=20000); r.add_argument("--nte", type=int, default=5000)
    r.add_argument("--epochs", type=int, default=12); r.add_argument("--bs", type=int, default=32)
    r.add_argument("--bert_lr", type=float, default=5e-5); r.add_argument("--head_lr", type=float, default=1e-3)
    r.add_argument("--seed", type=int, default=0); r.add_argument("--save", default=None, help="checkpoint dir")
    a = ap.parse_args(argv); t0 = time.time()
    smoke() if a.cmd == "smoke" else run(a)
    print(f"\n({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
