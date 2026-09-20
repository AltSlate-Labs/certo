"""Stage B training/eval: recover the known posterior through LANGUAGE with ModernBERT.

Same known-posterior world as v0, rendered to text (realize.py). Encoder reads state text; the
question stays a class partition applied to z (fixed universe; OUT = marginalization). Evaluates on
unseen states with SEEN wording and with UNSEEN (held-out) templates; every rendering of a given
state stays within one split (states never cross splits). Reports posterior fidelity (KL/TV) plus
calibration, and the invariance gates.

  python train_text.py smoke                       # tiny end-to-end wiring check (CPU/MPS ok)
  python train_text.py run --arm soft --device cuda --ntr 20000 --epochs 4
"""
from __future__ import annotations
import sys, argparse, time
import numpy as np, torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import datagen as D
import realize as R
from model import grouped_log_probs, arm_loss
from model_text import ModernBERTQueryModel, BACKBONE
from train import _partition


def build_text_dataset(corpus, n, seed, templates, tokenizer, max_len, world_qtypes=("choice", "binary", "score")):
    cfg = corpus.cfg; K = cfg.K; rng = np.random.default_rng(seed)
    texts = []
    r_full = np.zeros((n, K), np.float32); y = np.zeros(n, np.int64)
    G = np.zeros((n, K, K), np.float32); gmask = np.zeros((n, K), bool)
    for i in range(n):
        w = D.sample_world(corpus, rng)
        order = list(np.nonzero(w.reveal)[0]); rng.shuffle(order)
        t = int(rng.choice(templates))
        texts.append(R.render_state(cfg, w, t, order))
        r_full[i] = w.r; y[i] = w.y
        for g, cls in enumerate(_partition(world_qtypes[i % 3], K, rng)):
            G[i, g, cls] = 1.0; gmask[i, g] = True
    enc = tokenizer(texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    return dict(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                r_full=torch.from_numpy(r_full), y=torch.from_numpy(y),
                G=torch.from_numpy(G), gmask=torch.from_numpy(gmask))


def _batches(n, bs, gen=None):
    perm = torch.randperm(n, generator=gen)
    for s in range(0, n, bs):
        yield perm[s:s + bs]


def train(model, data, arm, epochs, bs, device, bert_lr, head_lr, seed=0, warmup=0.06):
    opt = torch.optim.AdamW(model.param_groups(bert_lr, head_lr))
    gen = torch.Generator().manual_seed(seed); n = data["y"].size(0); model.train()
    total = epochs * ((n + bs - 1) // bs)
    sched = get_linear_schedule_with_warmup(opt, int(warmup * total), total)
    for ep in range(epochs):
        tot = 0.0; nb = 0
        for b in _batches(n, bs, gen):
            z = model(data["input_ids"][b].to(device), data["attention_mask"][b].to(device))
            G = data["G"][b].to(device); gm = data["gmask"][b].to(device); r = data["r_full"][b].to(device)
            log_q = grouped_log_probs(z, G, gm)
            q_soft = (G * r.unsqueeze(1)).sum(2)
            loss = arm_loss(log_q, q_soft, gm, arm, r, G, gen)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item(); nb += 1
        print(f"  epoch {ep+1}/{epochs}  loss={tot/max(nb,1):.4f}  lr={sched.get_last_lr()[0]:.2e}", flush=True)
    return model


@torch.no_grad()
def evaluate(model, data, device, bs=256, nbins=15):
    model.eval(); zs = []
    for s in range(0, data["y"].size(0), bs):
        zs.append(model(data["input_ids"][s:s+bs].to(device),
                        data["attention_mask"][s:s+bs].to(device)).cpu())
    z = torch.cat(zs); p = torch.softmax(z, 1); r = data["r_full"]; y = data["y"]
    kl = (r * (r.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum(1).mean().item()
    tv = 0.5 * (r - p).abs().sum(1).mean().item()
    onehot = torch.zeros_like(p).scatter_(1, y[:, None], 1.0)
    brier = ((p - onehot) ** 2).sum(1).mean().item()
    acc = (p.argmax(1) == y).float().mean().item()
    conf = p.flatten(); hit = onehot.flatten(); bins = torch.clamp((conf * nbins).long(), 0, nbins - 1)
    ece = sum((bins == b).float().mean().item() * abs(conf[bins == b].mean() - hit[bins == b].mean()).item()
              for b in range(nbins) if (bins == b).any())
    log_q = grouped_log_probs(z, data["G"], data["gmask"])
    g_of_y = data["G"][torch.arange(y.size(0)), :, y].argmax(1)
    ans_acc = (log_q.argmax(1) == g_of_y).float().mean().item()
    return dict(KL=kl, TV=tv, Brier=brier, acc=acc, ECE=ece, ans_acc=ans_acc)


@torch.no_grad()
def invariance(model, data, device, K):
    """z is question-independent -> isolation exact; named-class odds independent of option set."""
    model.eval()
    z = model(data["input_ids"][:256].to(device), data["attention_mask"][:256].to(device)).cpu()
    rng = np.random.default_rng(0); max_err = 0.0
    for i in range(z.size(0)):
        a, b = (int(x) for x in rng.choice(K, size=2, replace=False))
        def odds(S):
            G = torch.zeros(1, K, K); gm = torch.zeros(1, K, dtype=torch.bool)
            for g, k in enumerate(S): G[0, g, k] = 1; gm[0, g] = True
            out = [k for k in range(K) if k not in S]
            if out:
                for k in out: G[0, len(S), k] = 1
                gm[0, len(S)] = True
            lq = grouped_log_probs(z[i:i+1], G, gm)
            return (lq[0, S.index(a)] - lq[0, S.index(b)]).item()
        S1 = [a, b]; S2 = S1 + [k for k in range(K) if k not in S1][:2]
        max_err = max(max_err, abs(odds(S1) - odds(S2)))
    return max_err


def _device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    return name


def run(a):
    dev = _device(a.device)
    print(f"Stage B (ModernBERT text)  world={a.world} arm={a.arm} device={dev} "
          f"Ntr={a.ntr} epochs={a.epochs} bs={a.bs} max_len={a.max_len}\n", flush=True)
    corpus = D.Corpus(D.ix_config() if a.world == "ix" else D.CorpusConfig()); K = corpus.cfg.K
    tok = AutoTokenizer.from_pretrained(a.backbone)
    tr = build_text_dataset(corpus, a.ntr, a.seed, R.TRAIN_TEMPLATES, tok, a.max_len)
    te_seen = build_text_dataset(corpus, a.nte, a.seed + 1, R.TRAIN_TEMPLATES, tok, a.max_len)
    te_new = build_text_dataset(corpus, a.nte, a.seed + 1, R.TEST_TEMPLATES, tok, a.max_len)
    torch.manual_seed(a.seed)
    model = ModernBERTQueryModel(K, a.backbone).to(dev)
    train(model, tr, a.arm, a.epochs, a.bs, dev, a.bert_lr, a.head_lr, a.seed)
    rs = evaluate(model, te_seen, dev); rn = evaluate(model, te_new, dev)
    inv = invariance(model, te_new, dev, K)
    print(f"\n{'eval':<22}{'KL(r|p)':>9}{'TV':>8}{'Brier':>8}{'acc':>7}{'ECE':>8}{'ans_acc':>9}")
    print("-" * 71)
    print(f"{'unseen states/seen wd':<22}{rs['KL']:>9.4f}{rs['TV']:>8.4f}{rs['Brier']:>8.4f}"
          f"{rs['acc']:>7.3f}{rs['ECE']:>8.4f}{rs['ans_acc']:>9.3f}")
    print(f"{'unseen states+templates':<22}{rn['KL']:>9.4f}{rn['TV']:>8.4f}{rn['Brier']:>8.4f}"
          f"{rn['acc']:>7.3f}{rn['ECE']:>8.4f}{rn['ans_acc']:>9.3f}")
    print(f"\n[{'PASS' if inv < 1e-3 else 'FAIL'}] odds-invariance (held-out templates)  max|Δlog-odds|={inv:.2e}")
    print("success = low KL/TV on BOTH rows (esp. unseen templates); invariance still passing.")


def smoke():
    """Tiny end-to-end wiring check; downloads ModernBERT-base on first run."""
    dev = _device("auto"); print("smoke device:", dev, flush=True)
    corpus = D.Corpus(D.CorpusConfig()); K = corpus.cfg.K
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    tr = build_text_dataset(corpus, 64, 0, R.TRAIN_TEMPLATES, tok, 160)
    te = build_text_dataset(corpus, 64, 1, R.TEST_TEMPLATES, tok, 160)
    model = ModernBERTQueryModel(K).to(dev)
    train(model, tr, "soft", 1, 16, dev, 2e-5, 1e-3)
    print("eval:", {k: round(v, 3) for k, v in evaluate(model, te, dev).items()})
    print("invariance max|Δlog-odds|:", f"{invariance(model, te, dev, K):.2e}")
    print("SMOKE OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description="JEV Stage B (ModernBERT text)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=["soft", "sampled", "argmax"], default="soft")
    r.add_argument("--world", choices=["v0", "ix"], default="v0")
    r.add_argument("--device", default="auto"); r.add_argument("--backbone", default=BACKBONE)
    r.add_argument("--ntr", type=int, default=20000); r.add_argument("--nte", type=int, default=5000)
    r.add_argument("--epochs", type=int, default=4); r.add_argument("--bs", type=int, default=32)
    r.add_argument("--max_len", type=int, default=192)
    r.add_argument("--bert_lr", type=float, default=2e-5); r.add_argument("--head_lr", type=float, default=1e-3)
    r.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.cmd == "smoke":
        smoke()
    else:
        run(a)
    print(f"\n({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
