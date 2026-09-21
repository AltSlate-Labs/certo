"""Test a certo checkpoint on the real JevBench public tasks (accuracy + Brier + ECE).

Maps each JevBench record (state + typed question with per-label criteria) to certo's decide(), reads
the calibrated probabilities over the exact label set. Not the full composite JevBench Score — a quick,
honest read on the public tasks. Full score = a certo_local adapter + their harness (later PR).

    python test_certo_jev.py <checkpoint_dir> <public_dir_or_jsonl>
"""
import sys, os, glob, json, math
sys.path.insert(0, os.path.expanduser("~/jev"))
import numpy as np, torch
from infer import DecisionModel

ckpt, tasks_path = sys.argv[1], sys.argv[2]
dm = DecisionModel.load(ckpt, device="cuda" if torch.cuda.is_available() else "cpu")


def load(p):
    files = [p] if p.endswith(".jsonl") else sorted(glob.glob(os.path.join(p, "*.jsonl")))
    return [(json.loads(l), os.path.basename(f)) for f in files for l in open(f)]


def options(t):
    q = t["question"]; crit = q.get("criteria")
    def desc(i, lbl):
        if isinstance(crit, dict) and lbl in crit:
            return str(crit[lbl])
        if isinstance(crit, list) and i < len(crit):
            return str(crit[i])
        if q.get("type") == "noul":
            return f"{q.get('instructions', '')} — {lbl}"
        return str(lbl)
    return [{"id": str(lbl), "description": desc(i, lbl)} for i, lbl in enumerate(t["labels"])]


from collections import defaultdict
by_file = defaultdict(lambda: dict(n=0, acc=0, brier=0.0, conf=[], hit=[], abst=0))
for t, fname in load(tasks_path):
    labels, exp = [str(x) for x in t["labels"]], str(t["expected"])
    state = t["state"] if isinstance(t["state"], str) else json.dumps(t["state"], ensure_ascii=False)
    r = dm.decide(state, options(t), max_state_len=256, max_option_len=64, abstain_below=0.5)
    p = r["probs"]; pred = max(p, key=p.get); s = by_file[fname]
    s["n"] += 1; s["acc"] += (pred == exp); s["abst"] += bool(r.get("abstain"))
    for lbl in labels:
        pv = float(p.get(lbl, 0.0)); h = 1.0 if lbl == exp else 0.0
        s["brier"] += (pv - h) ** 2; s["conf"].append(pv); s["hit"].append(h)


def ece(conf, hit, nb=15):
    conf, hit = np.array(conf), np.array(hit); b = np.clip((conf * nb).astype(int), 0, nb - 1)
    return sum((b == i).mean() * abs(conf[b == i].mean() - hit[b == i].mean()) for i in range(nb) if (b == i).any())


print(f"{'tier':<16}{'n':>6}{'acc':>8}{'brier':>9}{'ECE':>8}{'abstain':>9}")
for fname, s in sorted(by_file.items()):
    print(f"{fname:<16}{s['n']:>6}{s['acc']/s['n']:>8.3f}{s['brier']/s['n']:>9.4f}"
          f"{ece(s['conf'], s['hit']):>8.4f}{s['abst']/s['n']:>9.2f}")
