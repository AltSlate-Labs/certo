"""JEV synthetic data generator — conditional naive-Bayes evidence world (v0, symbolic).

Design: see DESIGN.md. Produces (state, question, options, exact-target) tuples where the
target is the analytically-known Bayes posterior, so we can score a model against the
Bayes-optimal answer (posterior fidelity), not just accuracy.

World:  likelihood tables theta_f(v|k) are FIXED across the corpus (frozen at build time, so
the hidden posterior is recoverable); the prior pi is drawn per world and WRITTEN into the
state; y ~ pi; each of F slots is revealed w.p. rho and emits v_f ~ theta_f(.|y). The posterior
is  P(y=k|E) ~ pi_k * prod_{revealed f} theta_f(v_f|k).

Questions over one shared state (all exact functions of the same posterior r):
  choice  named classes S (+ OUT = complement mass Sum_{k not in S} r_k when S subset of K)
  binary  predicate "y in B?" -> Sum_{k in B} r_k
  score   runtime class->level map -> mass per level; scalar = E[level]

Usage:
  python datagen.py selftest [--n N]          # run the verification suite (DESIGN.md sec 5)
  python datagen.py gen N OUT.jsonl [--seed S] # write N examples as JSONL
"""
from __future__ import annotations
import sys, json, math, argparse, itertools
from dataclasses import dataclass, asdict
import numpy as np

# --------------------------------------------------------------------------- corpus (frozen)

@dataclass
class CorpusConfig:
    K: int = 8              # classes (answer universe) -- fixed in v0
    F: int = 12             # feature slots
    V: int = 6              # values per slot
    prior_alpha: float = 0.7
    rho_range: tuple = (0.05, 0.80)      # per-world reveal rate -> entropy spread
    conc_log10_range: tuple = (-0.5, 1.5)  # per-slot Dirichlet conc -> discriminativeness
    n_noise: int = 2        # slots with identical rows across classes (pure distractors)
    n_pairs: int = 0        # interaction pairs: 2*n_pairs slots whose JOINT (not marginal) is informative
    pair_beta: float = 0.75 # pair joint strength (mix of a class-specific permutation vs uniform)
    prior_decimals: int = 3 # prior shown to this precision AND used for the target
    seed: int = 0


def ix_config(**kw) -> "CorpusConfig":
    """Interaction world: informative signal lives mostly in feature PAIRS (additive-model kryptonite)."""
    return CorpusConfig(**{"K": 8, "F": 15, "V": 6, "n_noise": 1, "n_pairs": 5, **kw})


class Corpus:
    """Frozen likelihood tables, deterministic in cfg.seed. Shared by every world."""
    def __init__(self, cfg: CorpusConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        conc = 10.0 ** rng.uniform(*cfg.conc_log10_range, size=cfg.F)  # low=peaked/strong
        theta = np.empty((cfg.F, cfg.K, cfg.V))
        for f in range(cfg.F):
            theta[f] = rng.dirichlet(np.full(cfg.V, conc[f]), size=cfg.K)
        for f in range(min(cfg.n_noise, cfg.F)):        # noise: same row for every class
            theta[f] = np.tile(rng.dirichlet(np.full(cfg.V, 1.0)), (cfg.K, 1))
        self.theta = theta
        with np.errstate(divide="ignore"):          # a 0-prob value legitimately => -inf (rules class out)
            self.logtheta = np.log(theta)
        self.conc = conc
        # slot allocation: [noise] [pairs...] [singles]. Pairs use 2 slots each.
        lo = min(cfg.n_noise, cfg.F)
        assert lo + 2 * cfg.n_pairs <= cfg.F, "F too small for n_noise + 2*n_pairs"
        self.pairs = [(lo + 2 * i, lo + 2 * i + 1) for i in range(cfg.n_pairs)]
        self.paired = {s for pr in self.pairs for s in pr}
        self.single_slots = [f for f in range(cfg.F) if f not in self.paired]
        # pair joint M[p,k] (V,V): doubly-stochastic/V -> uniform single-feature marginals for EVERY
        # class (so a lone member is uninformative), but a class-specific joint pattern.
        V = cfg.V; Mpair = np.empty((cfg.n_pairs, cfg.K, V, V))
        for p in range(cfg.n_pairs):
            for k in range(cfg.K):
                P = np.eye(V)[rng.permutation(V)]
                Mpair[p, k] = (cfg.pair_beta * P + (1 - cfg.pair_beta) * np.ones((V, V)) / V) / V
        self.Mpair = Mpair
        self.logMpair = np.log(Mpair) if cfg.n_pairs else np.zeros((0, cfg.K, V, V))

    @property
    def V(self): return self.cfg.V


# --------------------------------------------------------------------------- world sampling

@dataclass
class World:
    prior: np.ndarray   # (K,) quantized prior, == what the state text shows
    reveal: np.ndarray  # (F,) bool
    values: np.ndarray  # (F,) int, meaningful where reveal
    y: int
    rho: float
    r: np.ndarray       # (K,) exact posterior P(y|E)


def _logsumexp(a: np.ndarray) -> float:
    m = float(a.max())
    return m + math.log(float(np.exp(a - m).sum()))


def quantize(p: np.ndarray, decimals: int) -> np.ndarray:
    """Round to `decimals` and renormalize, so the displayed prior determines the target."""
    q = np.round(np.asarray(p, float), decimals)
    s = q.sum()
    if s <= 0:                       # pathological rounding -> fall back to raw
        q = np.asarray(p, float); s = q.sum()
    return q / s


def posterior(corpus: Corpus, prior: np.ndarray, reveal: np.ndarray, values: np.ndarray) -> np.ndarray:
    logp = np.log(prior + 1e-300)
    for f in corpus.single_slots:                       # single-feature evidence
        if reveal[f]:
            logp = logp + corpus.logtheta[f, :, values[f]]
    for p, (a, b) in enumerate(corpus.pairs):           # joint (interaction) evidence
        if reveal[a] and reveal[b]:
            logp = logp + corpus.logMpair[p, :, values[a], values[b]]
    return np.exp(logp - _logsumexp(logp))


def sample_world(corpus: Corpus, rng, fixed_prior=None, rho=None) -> World:
    cfg = corpus.cfg
    if fixed_prior is not None:
        prior = quantize(np.asarray(fixed_prior, float), cfg.prior_decimals)
    else:
        prior = quantize(rng.dirichlet(np.full(cfg.K, cfg.prior_alpha)), cfg.prior_decimals)
    if rho is None:
        rho = float(rng.uniform(*cfg.rho_range))
    y = int(rng.choice(cfg.K, p=prior))
    reveal = np.zeros(cfg.F, bool); values = np.full(cfg.F, -1, int)
    for f in corpus.single_slots:                       # single slots revealed independently
        if rng.random() < rho:
            reveal[f] = True; values[f] = int(rng.choice(cfg.V, p=corpus.theta[f, y]))
    for p, (a, b) in enumerate(corpus.pairs):           # a pair is revealed (and emitted) jointly
        if rng.random() < rho:
            reveal[a] = reveal[b] = True
            flat = int(rng.choice(cfg.V * cfg.V, p=corpus.Mpair[p, y].ravel()))
            values[a], values[b] = divmod(flat, cfg.V)
    r = posterior(corpus, prior, reveal, values)
    return World(prior, reveal, values, y, rho, r)


# --------------------------------------------------------------------------- entropy helpers

def dist_entropy(p) -> float:
    p = np.asarray(p, float); p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def bern_entropy(p: float) -> float:
    return dist_entropy([p, 1.0 - p]) if 0.0 < p < 1.0 else 0.0


# --------------------------------------------------------------------------- questions

def make_choice(world: World, rng, m: int) -> dict:
    """m named classes; add OUT (=complement mass) iff some classes are omitted."""
    K = len(world.r)
    m = min(m, K)
    S = rng.choice(K, size=m, replace=False)
    rng.shuffle(S)
    options = [{"tok": f"c{int(k)}", "class": int(k)} for k in S]
    target = [float(world.r[k]) for k in S]
    if m < K:                                   # OUT = raw complement, NOT renormalized
        options.append({"tok": "OUT", "class": None})
        target.append(float(1.0 - world.r[S].sum()))
    return {"type": "choice", "options": options, "target": target,
            "meta": {"entropy": dist_entropy(target), "has_out": m < K}}


def make_binary(world: World, rng) -> dict:
    K = len(world.r)
    B = rng.choice(K, size=int(rng.integers(1, K)), replace=False)   # 1..K-1 classes
    p = float(world.r[B].sum())
    return {"type": "binary", "subset": sorted(int(k) for k in B),
            "target": p, "meta": {"entropy": bern_entropy(p)}}


def make_score(world: World, rng, L: int) -> dict:
    K = len(world.r)
    c2l = rng.integers(0, L, size=K)            # runtime class->level map
    probs = np.zeros(L)
    np.add.at(probs, c2l, world.r)
    scalar = float((np.arange(L) * probs).sum())
    return {"type": "score", "levels": int(L), "class_to_level": [int(x) for x in c2l],
            "target": [float(x) for x in probs], "scalar": scalar,
            "meta": {"entropy": dist_entropy(probs)}}


def state_text(cfg: CorpusConfig, world: World, item_order) -> str:
    prior_str = " ".join(f"c{k}={world.prior[k]:.{cfg.prior_decimals}f}" for k in range(cfg.K))
    items = [f"f{f}=v{world.values[f]}" for f in item_order]
    return "prior: " + prior_str + " | " + (" | ".join(items) if items else "(no evidence)")


def build_example(corpus: Corpus, rng, world_id: int,
                  n_choice=1, n_binary=1, n_score=1,
                  m_range=(2, None), L_range=(3, 5)) -> dict:
    cfg = corpus.cfg
    w = sample_world(corpus, rng)
    order = list(np.nonzero(w.reveal)[0]); rng.shuffle(order)   # state is an unordered multiset
    qs = []
    mmax = m_range[1] or cfg.K
    for _ in range(n_choice):
        qs.append(make_choice(w, rng, int(rng.integers(m_range[0], mmax + 1))))
    for _ in range(n_binary):
        qs.append(make_binary(w, rng))
    for _ in range(n_score):
        qs.append(make_score(w, rng, int(rng.integers(L_range[0], L_range[1] + 1))))
    return {
        "world_id": world_id,
        "params": {"K": cfg.K, "F": cfg.F, "V": cfg.V, "rho": round(w.rho, 4),
                   "prior_alpha": cfg.prior_alpha},
        "prior": [round(float(x), cfg.prior_decimals) for x in w.prior],
        "state": [{"f": int(f), "v": int(w.values[f])} for f in order],
        "state_text": state_text(cfg, w, order),
        "y_true": w.y,                                  # accuracy + self-check only
        "posterior": [float(x) for x in w.r],           # full P(y|E), for eval
        "questions": qs,
    }


def generate(corpus: Corpus, n: int, seed: int):
    rng = np.random.default_rng(seed)
    for i in range(n):
        yield build_example(corpus, rng, world_id=i)


# --------------------------------------------------------------------------- verification (DESIGN.md sec 5)

def _wilson(k: int, n: int, z: float = 2.5758):     # ~99% CI, few false alarms across bins
    if n == 0:
        return (0.0, 1.0)
    phat = k / n; d = 1 + z * z / n
    c = (phat + z * z / (2 * n)) / d
    h = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def reliability_test(corpus: Corpus, n=50000, nbins=20, seed=1):
    """Oracle-label reliability: E[1(y=k)|r_k]=r_k is a theorem, so the curve must sit on the
    diagonal within a binomial CI. A violation => the posterior code is wrong."""
    rng = np.random.default_rng(seed)
    K = corpus.cfg.K
    conf = np.zeros(nbins); hit = np.zeros(nbins); cnt = np.zeros(nbins)
    for _ in range(n):
        w = sample_world(corpus, rng)
        b = np.clip((w.r * nbins).astype(int), 0, nbins - 1)
        np.add.at(conf, b, w.r)
        np.add.at(hit, b, (np.arange(K) == w.y).astype(float))
        np.add.at(cnt, b, 1.0)
    tot = cnt.sum(); ece = 0.0; viol = 0
    for b in range(nbins):
        if cnt[b] == 0:
            continue
        c, a = conf[b] / cnt[b], hit[b] / cnt[b]
        ece += cnt[b] / tot * abs(c - a)
        lo, hi = _wilson(int(hit[b]), int(cnt[b]))
        viol += not (lo - 1e-9 <= c <= hi + 1e-9)
    return {"ece": ece, "ci_violations": viol, "nbins_used": int((cnt > 0).sum())}


def tiny_world_exhaustive(seed=0):
    """Exact independent check on a tiny world: enumerate every full observation, confirm the
    generative distribution is proper (sum_E P(E)=1) and the closed form matches direct Bayes."""
    cfg = CorpusConfig(K=3, F=3, V=2, n_noise=0, seed=seed)
    corpus = Corpus(cfg)
    prior = quantize(np.random.default_rng(123).dirichlet(np.ones(3)), 6)
    reveal = np.ones(3, bool)
    total = 0.0; ok = True
    for vals in itertools.product(range(2), repeat=3):
        values = np.array(vals)
        joint = np.array([prior[k] * np.prod([corpus.theta[f, k, vals[f]] for f in range(3)])
                          for k in range(3)])
        total += joint.sum()
        direct = joint / joint.sum()
        post = posterior(corpus, prior, reveal, values)
        ok &= abs(post.sum() - 1) < 1e-9 and np.max(np.abs(direct - post)) < 1e-9
    return {"ok": bool(ok and abs(total - 1) < 1e-9), "sum_P_E": total}


def tiny_world_exhaustive_pairs(seed=0):
    """Same exact check for the interaction posterior: one pair, enumerate its joint values."""
    cfg = CorpusConfig(K=3, F=2, V=2, n_noise=0, n_pairs=1, pair_beta=0.7, seed=seed)
    c = Corpus(cfg)
    prior = quantize(np.random.default_rng(7).dirichlet(np.ones(3)), 6)
    reveal = np.ones(2, bool); total = 0.0; ok = True
    for va in range(2):
        for vb in range(2):
            values = np.array([va, vb])
            joint = np.array([prior[k] * c.Mpair[0, k, va, vb] for k in range(3)])
            total += joint.sum()
            post = posterior(c, prior, reveal, values)
            ok &= abs(post.sum() - 1) < 1e-9 and np.max(np.abs(joint / joint.sum() - post)) < 1e-9
    return {"ok": bool(ok and abs(total - 1) < 1e-9), "sum_P_E": total}


def unit_tests(corpus: Corpus, trials=3000, seed=2):
    rng = np.random.default_rng(seed); K = corpus.cfg.K; ok = True
    for _ in range(trials):
        w = sample_world(corpus, rng)
        for m in (2, min(3, K), K):
            q = make_choice(w, rng, m)
            ok &= abs(sum(q["target"]) - 1) < 1e-9         # normalized (incl. OUT)
        # named-class odds invariant to which other options appear; OUT mass grows with omission
        a, b = rng.choice(K, size=2, replace=False)
        small = np.array([a, b])
        big = np.unique(np.concatenate([small, rng.choice(K, size=min(4, K), replace=False)]))
        ra_s = _named_prob(make_choice_fixed(w, small), a); rb_s = _named_prob(make_choice_fixed(w, small), b)
        ra_b = _named_prob(make_choice_fixed(w, big), a);   rb_b = _named_prob(make_choice_fixed(w, big), b)
        ok &= abs(ra_s - ra_b) < 1e-12 and abs(rb_s - rb_b) < 1e-12   # raw posterior, not renormalized
        out_small = 1 - w.r[small].sum(); out_big = 1 - w.r[big].sum()
        ok &= out_small >= out_big - 1e-12
        # binary target == sum of the corresponding named choice masses (DESIGN sec 5)
        B = rng.choice(K, size=int(rng.integers(1, K)), replace=False)
        named = make_choice_fixed(w, B)
        choice_mass = sum(t for o, t in zip(named["options"], named["target"]) if o["class"] is not None)
        ok &= abs(float(w.r[B].sum()) - choice_mass) < 1e-12
        # score normalized + scalar consistent
        qs = make_score(w, rng, 4)
        ok &= abs(sum(qs["target"]) - 1) < 1e-9
        ok &= abs(qs["scalar"] - sum(i * p for i, p in enumerate(qs["target"]))) < 1e-9
    return {"ok": bool(ok)}


def make_choice_fixed(world: World, S: np.ndarray) -> dict:
    """Choice over a GIVEN named set S (no shuffling) -- for invariance unit tests."""
    K = len(world.r)
    options = [{"tok": f"c{int(k)}", "class": int(k)} for k in S]
    target = [float(world.r[k]) for k in S]
    if len(S) < K:
        options.append({"tok": "OUT", "class": None}); target.append(float(1 - world.r[S].sum()))
    return {"options": options, "target": target}


def _named_prob(q: dict, k: int) -> float:
    for o, t in zip(q["options"], q["target"]):
        if o["class"] == k:
            return t
    raise KeyError(k)


def mc_spot_check(corpus: Corpus, seed=3, trials=1_000_000):
    """Rejection-sampling check of one posterior: sample y'~prior, emit the revealed single slots,
    keep draws whose values match E; empirical class freq must match the closed form to within a
    few binomial SEs (tolerance scaled to the accepted count -- not a fixed threshold)."""
    rng = np.random.default_rng(seed)
    while True:                                   # single slots only, few reveals -> high acceptance
        w = sample_world(corpus, rng)
        idx = [f for f in np.nonzero(w.reveal)[0] if f in corpus.single_slots]
        if 1 <= len(idx) <= 2 and len(idx) == int(w.reveal.sum()):
            break
    K = corpus.cfg.K
    ys = rng.choice(K, size=trials, p=w.prior)
    match = np.ones(trials, bool)
    for f in idx:
        probs = corpus.theta[f, ys]
        v = (np.cumsum(probs, 1) > rng.random(trials)[:, None]).argmax(1)
        match &= (v == w.values[f])
    acc = int(match.sum())
    emp = np.bincount(ys[match], minlength=K) / max(acc, 1)
    se = np.sqrt(np.maximum(w.r * (1 - w.r), 1e-6) / max(acc, 1))
    return {"accepted": acc, "max_abs_dev": float(np.max(np.abs(emp - w.r))),
            "max_z": float(np.max(np.abs(emp - w.r) / se)) if acc else 99.0}


def entropy_coverage(corpus: Corpus, n=20000, seed=4):
    """Coverage judged by the entropy SPREAD (quantiles), not a single fixed bin. We want
    confident cases (low q10) AND genuinely uncertain ones (high q90). rho is the average knob,
    but a single conflicting item can raise per-example entropy, so we measure, not assume."""
    rng = np.random.default_rng(seed)
    H = np.empty(n); rhos = np.empty(n)
    for i in range(n):
        w = sample_world(corpus, rng); H[i] = dist_entropy(w.r); rhos[i] = w.rho
    Hmax = math.log(corpus.cfg.K)
    q10, q50, q90 = (np.quantile(H, q) / Hmax for q in (0.1, 0.5, 0.9))
    return {"q10": float(q10), "q50": float(q50), "q90": float(q90),
            "rho_H_corr": float(np.corrcoef(rhos, H)[0, 1]), "H_max_possible": Hmax}


def selftest(n=50000) -> bool:
    corpus = Corpus(CorpusConfig())
    print("JEV datagen self-test  (corpus:", asdict(corpus.cfg), ")\n")
    ok = True

    r = reliability_test(corpus, n=n)
    p = r["ci_violations"] <= 1 and r["ece"] < 0.01
    ok &= p; print(f"[{'PASS' if p else 'FAIL'}] oracle reliability   ece={r['ece']:.4f} "
                    f"ci_violations={r['ci_violations']}/{r['nbins_used']}")

    r = tiny_world_exhaustive()
    ok &= r["ok"]; print(f"[{'PASS' if r['ok'] else 'FAIL'}] tiny-world exact     "
                          f"sum_P(E)={r['sum_P_E']:.9f}")

    r = tiny_world_exhaustive_pairs()
    ok &= r["ok"]; print(f"[{'PASS' if r['ok'] else 'FAIL'}] tiny-world exact (pairs) "
                          f"sum_P(E)={r['sum_P_E']:.9f}")

    ixc = Corpus(ix_config())                     # interaction generator must also be calibrated
    ri = reliability_test(ixc, n=n)
    pi = ri["ci_violations"] <= 1 and ri["ece"] < 0.01
    ok &= pi; print(f"[{'PASS' if pi else 'FAIL'}] ix oracle reliability ece={ri['ece']:.4f} "
                    f"ci_violations={ri['ci_violations']}/{ri['nbins_used']}")

    r = unit_tests(corpus)
    ok &= r["ok"]; print(f"[{'PASS' if r['ok'] else 'FAIL'}] normalization/invariance unit tests")

    r = mc_spot_check(corpus)
    p = r["accepted"] > 300 and r["max_z"] < 5.0        # within ~5 binomial SE, count-scaled
    ok &= p; print(f"[{'PASS' if p else 'FAIL'}] MC posterior spot    accepted={r['accepted']} "
                    f"max_abs_dev={r['max_abs_dev']:.4f} max_z={r['max_z']:.2f}")

    r = entropy_coverage(corpus)
    p = r["q10"] < 0.25 and r["q90"] > 0.60 and r["rho_H_corr"] < 0     # spans confident->uncertain
    ok &= p; print(f"[{'PASS' if p else 'FAIL'}] entropy coverage     q10/50/90={r['q10']:.2f}/"
                    f"{r['q50']:.2f}/{r['q90']:.2f} of Hmax  corr(rho,H)={r['rho_H_corr']:.2f}")

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return ok


# --------------------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(description="JEV synthetic decision-data generator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("selftest"); st.add_argument("--n", type=int, default=50000)
    g = sub.add_parser("gen"); g.add_argument("n", type=int); g.add_argument("out")
    g.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    if a.cmd == "selftest":
        sys.exit(0 if selftest(a.n) else 1)
    if a.cmd == "gen":
        corpus = Corpus(CorpusConfig())
        with open(a.out, "w") as fh:
            for ex in generate(corpus, a.n, a.seed):
                fh.write(json.dumps(ex) + "\n")
        print(f"wrote {a.n} examples -> {a.out}")


if __name__ == "__main__":
    main()
