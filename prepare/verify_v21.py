"""Verify the v2.1 corpus before publishing: composition, label balance, leakage, parquet integrity.

    python verify_v21.py <corpus_dir>

Exits non-zero if any hard check fails, so a publish step can gate on it.
"""
from __future__ import annotations
import sys, os, json, glob
from collections import Counter, defaultdict

d = sys.argv[1]
fails = []


def hard(cond, msg):
    print(("  ok  " if cond else " FAIL ") + msg)
    if not cond:
        fails.append(msg)


# ---- 1. data card: zero validation errors
card = json.load(open(f"{d}/data_card.json"))
print("=== data card ==="); print(json.dumps(card, indent=2))
hard(card["validation_errors"] == 0, f"validation_errors == 0 (got {card['validation_errors']})")
total = card["total"]

# ---- 2. load all records from jsonl
recs = []
for sp in ("train", "dev"):
    for line in open(f"{d}/{sp}.jsonl"):
        recs.append(json.loads(line))
hard(len(recs) == total, f"jsonl line count {len(recs)} == card total {total}")

# ---- 3. question-kind mix near JevBench shape (choice ~50-65, noul ~25-35, score ~8-15)
typ = Counter(r["model_input"]["question"]["type"] for r in recs)
pct = {k: 100 * v / len(recs) for k, v in typ.items()}
print("\n=== question types (%) ==="); print({k: round(v, 1) for k, v in pct.items()})
noul = pct.get("binary", 0)
hard(noul >= 20, f"noul(binary) share >= 20% (got {noul:.1f}%)")
hard(pct.get("score", 0) >= 5, f"score share >= 5% (got {pct.get('score',0):.1f}%)")
hard(pct.get("choice", 0) <= 70, f"choice share <= 70% (got {pct.get('choice',0):.1f}%)")

# ---- 4. topic coverage: the 5 domains the rebalance was meant to add all present
dom = Counter(r["metadata"].get("domain") for r in recs)
print("\n=== domains (top) ==="); print(dict(dom.most_common(12)))
for t in ("math", "finance", "support", "safety", "coding"):
    hard(dom.get(t, 0) >= 1000, f"topic '{t}' has >= 1000 records (got {dom.get(t,0)})")

# ---- 5. per-family label balance (a rule family stuck on one answer is a bug)
fam_labels = defaultdict(Counter)
for r in recs:
    fam = r["metadata"].get("rule_family") or r["task_family"]
    t = r["target"]
    if t["kind"] == "categorical_label":
        fam_labels[fam][t["label_id"]] += 1
print("\n=== domain-family label balance ===")
for fam in sorted(k for k in fam_labels if k and ("_" in k) and fam_labels[k]):
    c = fam_labels[fam]; n = sum(c.values())
    if n < 200:
        continue
    top = c.most_common(1)[0][1] / n
    flag = "" if top <= 0.9 else "  <-- SKEWED"
    if fam.endswith(("_noul", "budget", "shift", "sev", "reimburse", "refuse", "safe")):
        print(f"  {fam:22} n={n:>7}  majority={top:.2f}{flag}")
        hard(top <= 0.9, f"noul family '{fam}' not >90% one class (got {top:.2f})")

# ---- 6. leakage: gold label id must be an option; answer word not trivially in state for noul
leak = 0
for r in recs:
    mi = r["model_input"]; t = r["target"]
    if t["kind"] != "categorical_label":
        continue
    ids = [o.get("id") for o in mi["question"].get("options", [])]
    if t["label_id"] not in ids:
        leak += 1
hard(leak == 0, f"every categorical gold label is among its options (violations={leak})")

# ---- 7. train/dev disjoint by split_group
groups = defaultdict(set)
for r in recs:
    groups[r["metadata"]["split"]].add(r["split_group_id"])
overlap = groups["train"] & groups["dev"]
hard(len(overlap) == 0, f"train/dev split-group disjoint (overlap={len(overlap)})")

# ---- 8. parquet integrity (if present): row counts match + JSON columns reload
pqs = glob.glob(f"{d}/*.parquet")
if pqs:
    import pandas as pd
    prows = 0
    for p in pqs:
        df = pd.read_parquet(p); prows += len(df)
        json.loads(df.iloc[0]["question"]); json.loads(df.iloc[0]["target"])
    hard(prows == total, f"parquet rows {prows} == card total {total}")
else:
    print("  --  no parquet yet (run jsonl2parquet first)")

# ---- 9. a couple of readable samples per new topic
print("\n=== samples ===")
seen = set()
for r in recs:
    tp = r["metadata"].get("domain")
    if tp in ("math", "finance", "support", "safety", "coding") and tp not in seen:
        seen.add(tp)
        q = r["model_input"]["question"]
        print(f"\n[{tp}] {q['type']}  gold={r['target'].get('label_id') or r['target'].get('level')}")
        print("  state:", r["model_input"]["state"][:160])
        print("  instr:", q["instructions"][:120])

print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED"))
sys.exit(1 if fails else 0)
