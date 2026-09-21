"""Milestone-1 build: a small mixed sample across families + output types, split, validated.

Covers the two fully-synthetic families (executable policies + known-posterior world) and all four
output types (choice / score / independent-binary / categorical-distribution). Public-dataset
adapters (MultiNLI, CLINC150, ...) are the next increment. Run from this directory:

    python build_sample.py [N] [OUT_DIR]
"""
from __future__ import annotations
import sys, os, json, hashlib
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # jev/ for genworld, realize
import numpy as np
import schema, policies
import genworld as G
from realize import ATTR_NAMES, VALUE_WORDS


def _state_text(cfg, e, rng):
    order = list(np.nonzero(e.reveal)[0]); rng.shuffle(order)
    if not order:
        return "No evidence has been observed."
    return "Observed evidence: " + "; ".join(f"{ATTR_NAMES[f]}={VALUE_WORDS[e.values[f]]}" for f in order) + "."


def _option_text(cfg, proto):
    return "typically " + ", ".join(f"{ATTR_NAMES[f]} {VALUE_WORDS[proto[f]]}" for f in range(cfg.F))


def posterior_records(n, seed=0):
    cfg = G.GenConfig(); rng = np.random.default_rng(seed); recs = []
    for i in range(n):
        e = G.sample_example(cfg, rng); M = e.protos.shape[0]
        opts = [{"id": f"opt{j}", "description": _option_text(cfg, e.protos[j])} for j in range(M)]
        q = {"type": "choice", "instructions": "Which option best matches the observed evidence?", "options": opts}
        probs = {f"opt{j}": float(e.r[j]) for j in range(M)}
        s = probs_total = sum(probs.values())
        probs = {k: v / s for k, v in probs.items()}                     # guard float drift < 1e-6
        sid = f"posterior-{i:06d}"
        recs.append(schema.record(f"{sid}-q0", sid, "known_posterior",
                    {"state": _state_text(cfg, e, rng), "question": q},
                    {"kind": "categorical_distribution", "probabilities": probs,
                     "source_type": "exact_posterior", "oracle_version": "genworld-v1"},
                    {"split": None, "source_id": "genworld", "rule_family": "match_count"}))
    return recs


def assign_split(r, frac=(0.8, 0.1, 0.1)):
    h = int(hashlib.md5(r["split_group_id"].encode()).hexdigest(), 16) / 2 ** 128
    r["metadata"]["split"] = "train" if h < frac[0] else ("dev" if h < frac[0] + frac[1] else "test")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/certo_v2_sample"
    os.makedirs(out, exist_ok=True)
    recs = policies.generate(n // 2, 0) + posterior_records(n // 2, 0)
    for r in recs:
        assign_split(r)

    errs = 0; by_err = Counter()
    for r in recs:
        for msg in schema.validate_record(r):
            errs += 1; by_err[msg] += 1

    buckets = {"train": [], "dev": [], "test": []}
    for r in recs:
        buckets[r["metadata"]["split"]].append(r)
    for sp, rs in buckets.items():
        with open(f"{out}/{sp}.jsonl", "w") as fh:
            for r in rs:
                fh.write(json.dumps(r) + "\n")

    fam = Counter(r["task_family"] for r in recs)
    typ = Counter(r["model_input"]["question"]["type"] for r in recs)
    prov = Counter(r["target"]["source_type"] for r in recs)
    sp = Counter(r["metadata"]["split"] for r in recs)
    print("=== certo v2 — Milestone-1 data sample ===")
    print(f"records: {len(recs)}  |  validation errors: {errs}")
    for m, c in by_err.items():
        print(f"  FAIL x{c}: {m}")
    print("families:    ", dict(fam))
    print("output types:", dict(typ))
    print("provenance:  ", dict(prov))
    print("splits:      ", dict(sp))
    print("\nexample model_input (choice):")
    ex = next(r for r in recs if r["task_family"].startswith("supplied_policy_threshold"))
    print(json.dumps(ex["model_input"], indent=1)[:600])
    print("target:", json.dumps(ex["target"]))
    print("\n" + ("ALL RECORDS VALID" if errs == 0 else "VALIDATION FAILURES ABOVE"))
    return 0 if errs == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
