"""Stage B text realizer: render a World (prior + revealed evidence) as natural-language surface.

The information content is EXACTLY the symbolic state (prior + which (feature,value) revealed) --
nothing about the hidden answer leaks (rendering never reads world.y). Several equivalent templates
let us hold some wording out for the "unseen templates" test; the underlying targets are unchanged,
so language variation is the only added difficulty (DESIGN.md sec 10, Stage B).

  python realize.py demo       # print a few rendered states across templates
  python realize.py selftest   # no-leakage + determinism + template-distinctness checks
"""
from __future__ import annotations
import sys
import numpy as np
import datagen as D

# Fixed vocab (>= the sizes we use). Class labels are the FIXED universe (OUT needs that).
CLASS_NAMES = ["amber", "cobalt", "crimson", "emerald", "indigo", "ochre", "teal", "violet",
               "saffron", "magenta"]
ATTR_NAMES = ["texture", "tempo", "density", "salinity", "cadence", "luster", "viscosity",
              "timbre", "gradient", "porosity", "torque", "humidity", "albedo", "turbidity",
              "resonance", "hue"]
VALUE_WORDS = ["frost", "dawn", "ember", "gale", "brine", "quartz", "slate", "cinder"]

TRAIN_TEMPLATES = (0, 1, 2)
TEST_TEMPLATES = (3,)                      # held-out wording


def _check_vocab(cfg):
    assert cfg.K <= len(CLASS_NAMES) and cfg.F <= len(ATTR_NAMES) and cfg.V <= len(VALUE_WORDS), \
        "vocab too small for this corpus config"


def _prior_str(cfg, prior, t: int) -> str:
    n = CLASS_NAMES
    if t == 0:
        return "Base rates: " + ", ".join(f"{n[k]} {prior[k]*100:.1f}%" for k in range(cfg.K)) + "."
    if t == 1:
        return "Prior probabilities — " + ", ".join(f"{n[k]}: {prior[k]:.3f}" for k in range(cfg.K)) + "."
    if t == 2:
        return "Historically, " + ", ".join(f"{n[k]} occurs {prior[k]*100:.1f}% of the time"
                                             for k in range(cfg.K)) + "."
    return "Baseline distribution over categories: " + \
           ", ".join(f"{n[k]}={prior[k]:.3f}" for k in range(cfg.K)) + "."


def _evidence_str(order, values, t: int) -> str:
    if len(order) == 0:
        return "No measurements are available."
    a, v = ATTR_NAMES, VALUE_WORDS
    if t == 0:
        return "Measurements: " + "; ".join(f"{a[f]}={v[values[f]]}" for f in order) + "."
    if t == 1:
        return " ".join(f"Observed {a[f]}: {v[values[f]]}." for f in order)
    if t == 2:
        return " ".join(f"The {a[f]} reading was {v[values[f]]}." for f in order)
    return "We measured " + ", ".join(f"{a[f]} as {v[values[f]]}" for f in order) + "."


def render_state(cfg, world, template: int, order) -> str:
    """Deterministic in (prior, order, values, template). Never reads world.y (=> no leakage)."""
    return _prior_str(cfg, world.prior, template) + " " + _evidence_str(order, world.values, template)


# --------------------------------------------------------------------------- checks

def selftest():
    cfg = D.CorpusConfig(); _check_vocab(cfg)
    corpus = D.Corpus(cfg); rng = np.random.default_rng(0)
    ok = True

    # 1) no leakage: rendering must not depend on the hidden class y
    import dataclasses
    for _ in range(2000):
        w = D.sample_world(corpus, rng)
        order = list(np.nonzero(w.reveal)[0]); rng.shuffle(order)
        w2 = dataclasses.replace(w, y=(w.y + 1) % cfg.K)     # different hidden answer, same evidence
        for t in TRAIN_TEMPLATES + TEST_TEMPLATES:
            if render_state(cfg, w, t, order) != render_state(cfg, w2, t, order):
                ok = False
    print(f"[{'PASS' if ok else 'FAIL'}] no-leakage (render independent of y)")

    # 2) determinism given order, and template distinctness
    w = D.sample_world(corpus, rng); order = list(np.nonzero(w.reveal)[0]); rng.shuffle(order)
    det = all(render_state(cfg, w, t, order) == render_state(cfg, w, t, order)
              for t in TRAIN_TEMPLATES + TEST_TEMPLATES)
    variants = {render_state(cfg, w, t, order) for t in TRAIN_TEMPLATES + TEST_TEMPLATES}
    distinct = len(variants) == len(TRAIN_TEMPLATES + TEST_TEMPLATES)
    ok &= det and distinct
    print(f"[{'PASS' if det else 'FAIL'}] deterministic given order")
    print(f"[{'PASS' if distinct else 'FAIL'}] templates produce distinct wording")

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return ok


def demo(n=2):
    cfg = D.CorpusConfig(); _check_vocab(cfg); corpus = D.Corpus(cfg); rng = np.random.default_rng(3)
    for _ in range(n):
        w = D.sample_world(corpus, rng); order = list(np.nonzero(w.reveal)[0]); rng.shuffle(order)
        print(f"(hidden y={CLASS_NAMES[w.y]}; posterior peak={CLASS_NAMES[int(w.r.argmax())]} "
              f"p={w.r.max():.2f})")
        for t in TRAIN_TEMPLATES + TEST_TEMPLATES:
            tag = "train" if t in TRAIN_TEMPLATES else "TEST "
            print(f"  [{tag} t{t}] {render_state(cfg, w, t, order)}")
        print()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    elif cmd == "demo":
        demo()
