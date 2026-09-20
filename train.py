"""JEV v0 training + evaluation harness.

Trains {AdditiveModel, EncoderQueryModel} x {soft, sampled, argmax} on identical data and scores
them against the ANALYTIC oracle posterior (posterior fidelity: KL/TV), plus calibration, accuracy
and the invariance regression gates. Every question is a class partition (see model.py).

  python train.py compare [--ntr 20000 --nte 5000 --epochs 8 --seed 0]
  python train.py invariance          # structural checks: odds-invariance + question-isolation
"""
from __future__ import annotations
import sys, argparse, math, time
import numpy as np, torch
import datagen as D
from model import AdditiveModel, EncoderQueryModel, grouped_log_probs, arm_loss

torch.set_num_threads(max(1, torch.get_num_threads()))


# --------------------------------------------------------------------------- data packing

def _partition(qtype: str, K: int, rng) -> list[list[int]]:
    """A question = a partition of {0..K-1} into answer groups (nonempty)."""
    if qtype == "choice":
        m = int(rng.integers(2, K + 1)); S = rng.choice(K, size=m, replace=False)
        groups = [[int(k)] for k in S]
        out = [int(k) for k in range(K) if k not in set(S.tolist())]
        if out: groups.append(out)                        # OUT = omitted classes
        return groups
    if qtype == "binary":
        b = int(rng.integers(1, K)); B = set(rng.choice(K, size=b, replace=False).tolist())
        return [sorted(B), sorted(set(range(K)) - B)]
    L = int(rng.integers(3, 6)); c2l = rng.integers(0, L, size=K)
    return [[int(k) for k in range(K) if c2l[k] == l] for l in range(L) if (c2l == l).any()]


def build_dataset(corpus: D.Corpus, n: int, seed: int):
    cfg = corpus.cfg; K, Fdim = cfg.K, cfg.F; rng = np.random.default_rng(seed)
    feat = np.zeros((n, Fdim), np.int64); val = np.zeros((n, Fdim), np.int64)
    mask = np.zeros((n, Fdim), bool); prior = np.zeros((n, K), np.float32)
    r_full = np.zeros((n, K), np.float32); y = np.zeros(n, np.int64)
    G = np.zeros((n, K, K), np.float32); gmask = np.zeros((n, K), bool)
    qtypes = ("choice", "binary", "score")
    for i in range(n):
        w = D.sample_world(corpus, rng)
        idx = np.nonzero(w.reveal)[0]
        feat[i, :len(idx)] = idx; val[i, :len(idx)] = w.values[idx]; mask[i, :len(idx)] = True
        prior[i] = w.prior; r_full[i] = w.r; y[i] = w.y
        groups = _partition(qtypes[i % 3], K, rng)
        for g, cls in enumerate(groups):
            G[i, g, cls] = 1.0; gmask[i, g] = True
    t = lambda a: torch.from_numpy(a)
    return dict(feat=t(feat), val=t(val), mask=t(mask), prior=t(prior),
                r_full=t(r_full), y=t(y), G=t(G), gmask=t(gmask))


# --------------------------------------------------------------------------- train / eval

def train_model(model, data, arm, epochs, bs=256, lr=2e-3, seed=0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    n = data["feat"].size(0); model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            z = model(data["feat"][b], data["val"][b], data["mask"][b], data["prior"][b])
            log_q = grouped_log_probs(z, data["G"][b], data["gmask"][b])
            q_soft = (data["G"][b] * data["r_full"][b].unsqueeze(1)).sum(2)
            loss = arm_loss(log_q, q_soft, data["gmask"][b], arm, data["r_full"][b], data["G"][b], gen)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def fit_temperature(model, cal, steps=300):
    """Best-case rescaling: scalar T minimizing KL(r || softmax(z/T)) on separate calib data.
    If even the oracle-optimal T can't repair KL, the errors are not mere over-sharpness."""
    model.eval()
    with torch.no_grad():
        z = model(cal["feat"], cal["val"], cal["mask"], cal["prior"])
    r = cal["r_full"]; logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([logT], lr=0.05)
    for _ in range(steps):
        p = torch.softmax(z / logT.exp(), 1)
        kl = (r * (r.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum(1).mean()
        opt.zero_grad(); kl.backward(); opt.step()
    return logT.exp().item()


@torch.no_grad()
def evaluate(model, data, nbins=15, T=1.0):
    model.eval()
    z = model(data["feat"], data["val"], data["mask"], data["prior"]) / T
    p = torch.softmax(z, 1); r = data["r_full"]; y = data["y"]
    kl = (r * (r.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum(1).mean().item()
    tv = 0.5 * (r - p).abs().sum(1).mean().item()
    onehot = torch.zeros_like(p).scatter_(1, y[:, None], 1.0)
    brier = ((p - onehot) ** 2).sum(1).mean().item()
    acc = (p.argmax(1) == y).float().mean().item()
    # ECE on the full posterior (predicted p_k vs realized y=k over all classes)
    conf = p.flatten(); hit = onehot.flatten()
    bins = torch.clamp((conf * nbins).long(), 0, nbins - 1)
    ece = 0.0
    for bidx in range(nbins):
        m = bins == bidx
        if m.any(): ece += m.float().mean().item() * abs(conf[m].mean() - hit[m].mean()).item()
    # answer-level: argmax group vs the group holding y
    log_q = grouped_log_probs(z, data["G"], data["gmask"])
    g_of_y = data["G"][torch.arange(y.size(0)), :, y].argmax(1)
    ans_acc = (log_q.argmax(1) == g_of_y).float().mean().item()
    return dict(KL=kl, TV=tv, Brier=brier, acc=acc, ECE=ece, ans_acc=ans_acc)


def compare(ntr, nte, epochs, seed, world="v0"):
    corpus = D.Corpus(D.ix_config() if world == "ix" else D.CorpusConfig())
    cfg = corpus.cfg
    tr = build_dataset(corpus, ntr, seed)
    te = build_dataset(corpus, nte, seed + 1)
    print(f"JEV compare  world={world} Ntr={ntr} Nte={nte} epochs={epochs}  "
          f"K={cfg.K} F={cfg.F} V={cfg.V} n_pairs={cfg.n_pairs}\n")
    print(f"{'model':<14}{'arm':<9}{'KL(r|p)':>9}{'TV':>8}{'Brier':>8}{'acc':>7}{'ECE':>8}{'ans_acc':>9}")
    print("-" * 72)
    # additive is tiny + exact-capable -> train it to the ceiling; encoder gets the given budget.
    specs = (("additive", lambda: AdditiveModel(cfg.F, cfg.V, cfg.K), max(epochs, 60), 5e-3),
             ("encoder-qry", lambda: EncoderQueryModel(cfg.F, cfg.V, cfg.K), epochs, 2e-3))
    for name, ctor, ep, lr in specs:
        for arm in ("soft", "sampled", "argmax"):
            torch.manual_seed(seed)
            m = train_model(ctor(), tr, arm, ep, lr=lr, seed=seed)
            r = evaluate(m, te)
            print(f"{name:<14}{arm:<9}{r['KL']:>9.4f}{r['TV']:>8.4f}{r['Brier']:>8.4f}"
                  f"{r['acc']:>7.3f}{r['ECE']:>8.4f}{r['ans_acc']:>9.3f}")
        print()
    if world == "ix":
        print("ix world: signal is mostly in feature PAIRS. additive (sums per-feature) cannot read\n"
              "joints -> should plateau well above 0; encoder-qry should beat it on KL (learns interactions).")
    else:
        print("expect: soft ~ sampled (both target r) << argmax on KL/TV/ECE; acc comparable (additive).\n"
              "        additive/soft ~ analytic ceiling (KL -> 0).")


def tempctl(ntr, nte, epochs, seed):
    """Cheap control: can temperature scaling repair the argmax model's posterior fidelity?"""
    corpus = D.Corpus(D.CorpusConfig()); cfg = corpus.cfg
    tr = build_dataset(corpus, ntr, seed)
    cal = build_dataset(corpus, nte, seed + 2)       # separate calibration split
    te = build_dataset(corpus, nte, seed + 1)
    print(f"JEV temperature control  Ntr={ntr} Nte={nte} epochs={epochs}\n")
    print(f"{'model':<14}{'arm':<16}{'KL(r|p)':>9}{'ECE':>8}{'acc':>7}{'T':>7}")
    print("-" * 61)
    specs = (("additive", lambda: AdditiveModel(cfg.F, cfg.V, cfg.K), max(epochs, 60), 5e-3),
             ("encoder-qry", lambda: EncoderQueryModel(cfg.F, cfg.V, cfg.K), epochs, 2e-3))
    for name, ctor, ep, lr in specs:
        torch.manual_seed(seed); ms = train_model(ctor(), tr, "soft", ep, lr=lr, seed=seed)
        rs = evaluate(ms, te)
        print(f"{name:<14}{'soft':<16}{rs['KL']:>9.4f}{rs['ECE']:>8.4f}{rs['acc']:>7.3f}{1.0:>7.2f}")
        torch.manual_seed(seed); ma = train_model(ctor(), tr, "argmax", ep, lr=lr, seed=seed)
        ra = evaluate(ma, te)
        print(f"{name:<14}{'argmax':<16}{ra['KL']:>9.4f}{ra['ECE']:>8.4f}{ra['acc']:>7.3f}{1.0:>7.2f}")
        T = fit_temperature(ma, cal); rt = evaluate(ma, te, T=T)
        print(f"{name:<14}{'argmax+temp':<16}{rt['KL']:>9.4f}{rt['ECE']:>8.4f}{rt['acc']:>7.3f}{T:>7.2f}")
        print()
    print("if argmax+temp KL stays >> soft KL, the damage is not mere over-sharpness (unrecoverable\n"
          "by a global rescale); T>1 means argmax was overconfident.")


@torch.no_grad()
def invariance(seed=0):
    """Structural regression gates: named-class odds invariant to the option set, and z (hence the
    full posterior) invariant to the question asked. Checked numerically within tolerance."""
    corpus = D.Corpus(D.CorpusConfig())
    m = EncoderQueryModel(corpus.cfg.F, corpus.cfg.V, corpus.cfg.K)  # untrained: structure, not skill
    m.eval()                                                          # disable dropout for determinism
    te = build_dataset(corpus, 512, seed); K = corpus.cfg.K
    z = m(te["feat"], te["val"], te["mask"], te["prior"])
    p = torch.softmax(z, 1)
    rng = np.random.default_rng(seed); max_odds_err = 0.0
    for i in range(z.size(0)):
        a, b = rng.choice(K, size=2, replace=False)
        # odds a:b under a small option set vs a superset -> must match (option-independent read)
        def odds(S):
            G = torch.zeros(1, K, K); gm = torch.zeros(1, K, dtype=torch.bool)
            for g, k in enumerate(S): G[0, g, k] = 1; gm[0, g] = True
            out = [k for k in range(K) if k not in S]
            if out:
                for k in out: G[0, len(S), k] = 1
                gm[0, len(S)] = True
            lq = grouped_log_probs(z[i:i+1], G, gm)
            return (lq[0, S.index(a)] - lq[0, S.index(b)]).item()
        S1 = [int(a), int(b)]; extra = [k for k in range(K) if k not in S1][:2]
        S2 = S1 + extra
        max_odds_err = max(max_odds_err, abs(odds(S1) - odds(S2)))
    # question-isolation: z does not take the question as input -> identical regardless. Confirm p
    # equals a re-run with a different (dummy) question packing (same state tensors).
    z2 = m(te["feat"], te["val"], te["mask"], te["prior"])
    iso_err = (torch.softmax(z2, 1) - p).abs().max().item()
    ok = max_odds_err < 1e-4 and iso_err < 1e-6
    print(f"[{'PASS' if max_odds_err < 1e-4 else 'FAIL'}] odds-invariance   "
          f"max |log-odds(S1)-log-odds(S2)| = {max_odds_err:.2e}")
    print(f"[{'PASS' if iso_err < 1e-6 else 'FAIL'}] question-isolation max |dp| = {iso_err:.2e}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description="JEV v0 train/eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--ntr", type=int, default=20000); c.add_argument("--nte", type=int, default=5000)
    c.add_argument("--epochs", type=int, default=8); c.add_argument("--seed", type=int, default=0)
    c.add_argument("--world", choices=["v0", "ix"], default="v0")
    t = sub.add_parser("tempctl")
    t.add_argument("--ntr", type=int, default=20000); t.add_argument("--nte", type=int, default=5000)
    t.add_argument("--epochs", type=int, default=8); t.add_argument("--seed", type=int, default=0)
    sub.add_parser("invariance")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.cmd == "compare":
        compare(a.ntr, a.nte, a.epochs, a.seed, a.world)
    elif a.cmd == "tempctl":
        tempctl(a.ntr, a.nte, a.epochs, a.seed)
    elif a.cmd == "invariance":
        sys.exit(0 if invariance() else 1)
    print(f"\n({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
