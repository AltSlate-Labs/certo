"""Generic decision world (Stage C / step 1): options are RUNTIME, defined by a description.

Toward a generic decision primitive (state + runtime options -> calibrated typed answer), we drop
the fixed class universe. Each option is a *prototype*: "what evidence this option typically
produces." Given the observed state, an option's exact probability is a Bayesian match between the
evidence and its stated prototype -- so options are fully runtime, any number can be presented, and
an UNSEEN option is just a new prototype. Still 100% known-answer (exact posterior), so gradeable.

Generative model (per example):
  - present M options; option o has prototype proto[o] in {0..V-1}^F  (drawn at runtime).
  - true option y ~ uniform(M); each feature revealed w.p. rho; a revealed feature emits
        v_f = proto[y,f]        with prob 1-eps
        v_f ~ uniform(V)        with prob eps        (noise)
    so  theta(v | o,f) = (1-eps)*1[v==proto[o,f]] + eps/V.
  - posterior over the presented options:
        P(y=o | E) ∝ prod_{revealed f} theta(v_f | o,f)      (uniform prior drops out)
    i.e. a match-count: log-lik = #match*log((1-eps)+eps/V) + #mismatch*log(eps/V).

The model must learn the RULE (match observed evidence to each option's described prototype), not
fixed class signatures -> generalizes to unseen options / new option counts.

  python genworld.py selftest
"""
from __future__ import annotations
import sys, math, itertools
from dataclasses import dataclass
import numpy as np


@dataclass
class GenConfig:
    F: int = 6            # features
    V: int = 4            # values per feature
    eps: float = 0.15     # emission noise
    M_range: tuple = (3, 6)      # number of options presented
    rho_range: tuple = (0.2, 0.95)   # per-example feature-reveal rate
    seed: int = 0


@dataclass
class GenExample:
    protos: np.ndarray    # (M, F) option prototypes
    reveal: np.ndarray    # (F,) bool
    values: np.ndarray    # (F,) int  (meaningful where reveal)
    y: int                # index of the true option
    r: np.ndarray         # (M,) exact posterior over presented options


def _logsumexp(a):
    m = float(a.max()); return m + math.log(float(np.exp(a - m).sum()))


def posterior(protos, reveal, values, eps, V):
    M = protos.shape[0]
    a = math.log((1 - eps) + eps / V)      # log-lik of a matching feature
    b = math.log(eps / V)                  # log-lik of a mismatching feature
    idx = np.nonzero(reveal)[0]
    logp = np.zeros(M)
    for o in range(M):
        match = (protos[o, idx] == values[idx])
        logp[o] = match.sum() * a + (~match).sum() * b
    return np.exp(logp - _logsumexp(logp))


def sample_example(cfg, rng, M=None, rho=None) -> GenExample:
    M = M or int(rng.integers(cfg.M_range[0], cfg.M_range[1] + 1))
    rho = rho if rho is not None else float(rng.uniform(*cfg.rho_range))
    protos = rng.integers(0, cfg.V, size=(M, cfg.F))
    y = int(rng.integers(M))
    reveal = rng.random(cfg.F) < rho
    values = np.full(cfg.F, -1, int)
    for f in np.nonzero(reveal)[0]:
        values[f] = int(protos[y, f]) if rng.random() < 1 - cfg.eps else int(rng.integers(cfg.V))
    r = posterior(protos, reveal, values, cfg.eps, cfg.V)
    return GenExample(protos, reveal, values, y, r)


def dist_entropy(p):
    p = np.asarray(p, float); p = p[p > 0]; return float(-(p * np.log(p)).sum())


# --------------------------------------------------------------------------- verification

def _wilson(k, n, z=2.5758):
    if n == 0: return (0.0, 1.0)
    ph = k / n; d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def reliability_test(cfg, n=60000, nbins=20, seed=1):
    """Oracle labels must be calibrated: bin predicted r_o, empirical P(y=o) lies on the diagonal
    within a binomial CI (a theorem if the posterior code is right)."""
    rng = np.random.default_rng(seed)
    conf = np.zeros(nbins); hit = np.zeros(nbins); cnt = np.zeros(nbins)
    for _ in range(n):
        e = sample_example(cfg, rng)
        b = np.clip((e.r * nbins).astype(int), 0, nbins - 1)
        np.add.at(conf, b, e.r); np.add.at(cnt, b, 1.0)
        onehot = np.zeros(len(e.r)); onehot[e.y] = 1.0
        np.add.at(hit, b, onehot)
    tot = cnt.sum(); ece = 0.0; viol = 0
    for b in range(nbins):
        if cnt[b] == 0: continue
        c, ac = conf[b] / cnt[b], hit[b] / cnt[b]; ece += cnt[b] / tot * abs(c - ac)
        lo, hi = _wilson(int(hit[b]), int(cnt[b])); viol += not (lo - 1e-9 <= c <= hi + 1e-9)
    return {"ece": ece, "ci_violations": viol, "nbins_used": int((cnt > 0).sum())}


def tiny_exhaustive(seed=0):
    """Exact check on a tiny world: fixed prototypes, enumerate all full observations."""
    cfg = GenConfig(F=3, V=2, eps=0.2, seed=seed)
    rng = np.random.default_rng(5); protos = rng.integers(0, 2, size=(2, 3))
    reveal = np.ones(3, bool); total = 0.0; ok = True
    a = (1 - cfg.eps) + cfg.eps / cfg.V; b = cfg.eps / cfg.V
    for vals in itertools.product(range(2), repeat=3):
        values = np.array(vals)
        # P(E) = sum_y (1/M) * prod_f theta(v_f|y)  -- but here uniform y and emission define joint;
        # check posterior is a proper distribution and matches the direct match-count formula.
        lik = np.array([np.prod([a if protos[o, f] == vals[f] else b for f in range(3)]) for o in range(2)])
        direct = lik / lik.sum()
        post = posterior(protos, reveal, values, cfg.eps, cfg.V)
        ok &= abs(post.sum() - 1) < 1e-9 and np.max(np.abs(direct - post)) < 1e-9
        total += lik.sum()
    return {"ok": bool(ok)}       # (total is not 1 here: lik is per-option, not a joint over E)


def unit_tests(cfg, trials=5000, seed=2):
    rng = np.random.default_rng(seed); ok = True
    for _ in range(trials):
        e = sample_example(cfg, rng)
        ok &= abs(e.r.sum() - 1) < 1e-9 and e.r.min() >= -1e-12
        # revealing zero features -> uniform posterior over the M options
    e0 = sample_example(cfg, np.random.default_rng(9), rho=0.0)
    ok &= np.allclose(e0.r, np.ones(len(e0.r)) / len(e0.r))
    return {"ok": bool(ok)}


def entropy_coverage(cfg, n=20000, seed=4):
    rng = np.random.default_rng(seed); H = []
    for _ in range(n):
        e = sample_example(cfg, rng); H.append(dist_entropy(e.r) / math.log(len(e.r)))
    H = np.array(H)
    q10, q50, q90 = (float(np.quantile(H, q)) for q in (0.1, 0.5, 0.9))
    return {"q10": q10, "q50": q50, "q90": q90}      # fraction of max entropy (log M)


def selftest(n=60000):
    cfg = GenConfig()
    print("genworld self-test  cfg:", cfg, "\n")
    ok = True
    r = reliability_test(cfg, n=n)
    p = r["ci_violations"] <= 1 and r["ece"] < 0.01; ok &= p
    print(f"[{'PASS' if p else 'FAIL'}] oracle reliability   ece={r['ece']:.4f} ci_violations={r['ci_violations']}/{r['nbins_used']}")
    r = tiny_exhaustive(); ok &= r["ok"]
    print(f"[{'PASS' if r['ok'] else 'FAIL'}] tiny-world exact")
    r = unit_tests(cfg); ok &= r["ok"]
    print(f"[{'PASS' if r['ok'] else 'FAIL'}] normalization + no-evidence=uniform")
    r = entropy_coverage(cfg)
    p = r["q10"] < 0.25 and r["q90"] > 0.6; ok &= p
    print(f"[{'PASS' if p else 'FAIL'}] entropy coverage     q10/50/90={r['q10']:.2f}/{r['q50']:.2f}/{r['q90']:.2f} of log(M)")
    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if selftest() else 1)
