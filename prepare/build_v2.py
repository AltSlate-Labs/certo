"""Build the v2 real-data corpus (>=1M) in the canonical decision schema.

Gold public datasets (real human labels) + our synthetic (exact-posterior + executable policies),
each adapted to one schema with explicit provenance. Only source TRAIN splits are used; official
validation/test are left untouched. Project split (train/dev) is by hashed group id.

    python build_v2.py <out_dir> [--tiny]      # --tiny caps every source to 300 for a pipeline check
"""
from __future__ import annotations
import sys, os, json, hashlib, itertools, re
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # jev/ for genworld
import numpy as np
from datasets import load_dataset, get_dataset_config_info
import schema, policies, domains
import genworld as G
from realize import ATTR_NAMES, VALUE_WORDS

TINY = "--tiny" in sys.argv
CAP = 300 if TINY else None


def human(s):
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(s))
    return s.replace("_", " ").strip().lower()


def label_names(name, cfg, field):
    f = get_dataset_config_info(name, cfg).features[field]
    return list(getattr(f, "names", None) or f.feature.names)   # handles ClassLabel and Sequence(ClassLabel)


def candidate_ids(gold, all_ids, k, rng):
    negs = [i for i in all_ids if i != gold]
    rng.shuffle(negs)
    sub = [gold] + negs[:k]
    rng.shuffle(sub)
    return sub


def rec(rid, sid, fam, state, q, target, meta):
    return schema.record(f"{rid}", sid, fam, {"state": state, "question": q}, target,
                         {"split": None, **meta})


# --------------------------------------------------------------------------- gold adapters

def dbpedia(cap, rng):
    names = label_names("fancyzhx/dbpedia_14", None, "label")
    opts = [{"id": f"c{i}", "description": human(n)} for i, n in enumerate(names)]
    ds = load_dataset("fancyzhx/dbpedia_14", split="train", streaming=True)
    for i, r in enumerate(itertools.islice(ds, cap or 10**9)):
        q = {"type": "choice", "instructions": "Which category does this text belong to?", "options": opts}
        yield rec(f"dbpedia-{i}", f"dbpedia-{i}", "topic_classification", r["content"].strip(), q,
                  {"kind": "categorical_label", "label_id": f"c{r['label']}", "source_type": "human_annotation"},
                  {"source_id": "fancyzhx/dbpedia_14", "license": "cc-by-sa-3.0", "domain": "wikipedia"})


def multinli(cap, rng):
    opts = [{"id": "entailment", "description": "the premise supports the statement"},
            {"id": "neutral", "description": "the premise neither supports nor contradicts it"},
            {"id": "contradiction", "description": "the premise contradicts the statement"}]
    ids = ["entailment", "neutral", "contradiction"]
    ds = load_dataset("nyu-mll/multi_nli", split="train", streaming=True)
    for i, r in enumerate(itertools.islice(ds, cap or 10**9)):
        if r["label"] < 0:
            continue
        q = {"type": "choice", "instructions": f"Given the premise, is this statement supported? "
             f"Statement: \"{r['hypothesis'].strip()}\"", "options": opts}
        yield rec(f"mnli-{i}", f"mnli-{i}", "evidence_inference", r["premise"].strip(), q,
                  {"kind": "categorical_label", "label_id": ids[r["label"]], "source_type": "human_annotation"},
                  {"source_id": "nyu-mll/multi_nli", "license": "mixed-see-source", "genre": r.get("genre")})


def goemotions(cap, rng):
    names = label_names("google-research-datasets/go_emotions", "simplified", "labels")
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="train", streaming=True)
    alln = list(range(len(names)))
    for i, r in enumerate(itertools.islice(ds, cap or 10**9)):
        pos = list(r["labels"])
        negs = [x for x in alln if x not in pos]; rng.shuffle(negs)
        present = pos + negs[:max(6, 10 - len(pos))]
        rng.shuffle(present)
        opts = [{"id": names[x], "description": f"expresses {human(names[x])}"} for x in present]
        vals = {names[x]: (1 if x in pos else 0) for x in present}
        q = {"type": "independent_binary", "instructions": "Which emotions does this text express?", "options": opts}
        yield rec(f"goemo-{i}", f"goemo-{i}", "emotion_multilabel", r["text"].strip(), q,
                  {"kind": "bernoulli_labels", "values": vals, "observed_mask": {k: True for k in vals},
                   "source_type": "human_annotation"},
                  {"source_id": "google-research-datasets/go_emotions", "license": "apache-2.0"})


def clinc(cap, rng):
    names = label_names("clinc/clinc_oos", "plus", "intent")
    allids = list(range(len(names)))
    ds = load_dataset("clinc/clinc_oos", "plus", split="train", streaming=True)
    for i, r in enumerate(itertools.islice(ds, cap or 10**9)):
        gold = r["intent"]
        sub = candidate_ids(gold, allids, 7, rng)
        opts = [{"id": names[x], "description": human(names[x])} for x in sub]
        q = {"type": "choice", "instructions": "Which intent best matches the request?", "options": opts}
        yield rec(f"clinc-{i}", f"clinc-{i}", "intent_classification", r["text"].strip(), q,
                  {"kind": "categorical_label", "label_id": names[gold], "source_type": "human_annotation"},
                  {"source_id": "clinc/clinc_oos", "license": "cc-by-3.0", "candidate_policy": "gold+7random"})


def boolq(cap, rng):  # noul: a single yes/no read against a passage
    opts = [{"id": "yes", "description": "the answer is yes"}, {"id": "no", "description": "the answer is no"}]
    ds = load_dataset("google/boolq", split="train", streaming=True)
    for i, r in enumerate(itertools.islice(ds, cap or 10**9)):
        q = {"type": "binary", "options": opts,
             "instructions": f"Based on the passage, answer yes or no: {r['question'].strip()}?"}
        yield rec(f"boolq-{i}", f"boolq-{i}", "boolean_qa", r["passage"].strip(), q,
                  {"kind": "categorical_label", "label_id": "yes" if r["answer"] else "no",
                   "source_type": "human_annotation"},
                  {"source_id": "google/boolq", "license": "cc-by-sa-3.0"})


def arc(cap, rng):
    for conf in ["ARC-Easy", "ARC-Challenge"]:
        ds = load_dataset("allenai/ai2_arc", conf, split="train", streaming=True)
        for i, r in enumerate(itertools.islice(ds, (cap or 10**9))):
            texts, labs = r["choices"]["text"], r["choices"]["label"]
            opts = [{"id": labs[j], "description": texts[j]} for j in range(len(texts))]
            if r["answerKey"] not in labs:
                continue
            q = {"type": "choice", "instructions": r["question"].strip(), "options": opts}
            yield rec(f"{conf}-{i}", f"{conf}-{i}", "reasoning_mcq", r["question"].strip(), q,
                      {"kind": "categorical_label", "label_id": r["answerKey"], "source_type": "human_annotation"},
                      {"source_id": "allenai/ai2_arc", "config": conf, "license": "cc-by-sa-4.0"})


# --------------------------------------------------------------------------- synthetic adapters

def synth_policies(cap, rng):
    for r in policies.generate(cap or 100000, seed=7):
        r["metadata"]["split"] = None
        yield r


def synth_domains(cap, rng):  # JevBench-representative math/finance/ops/safety/coding decisions
    for r in domains.generate(cap or 500000, seed=11):
        r["metadata"]["split"] = None
        yield r


def synth_posterior(cap, rng):
    cfg = G.GenConfig(); r2 = np.random.default_rng(3)
    for i in range(cap or 60000):
        e = G.sample_example(cfg, r2)
        order = list(np.nonzero(e.reveal)[0]); r2.shuffle(order)
        M = e.protos.shape[0]
        opts = [{"id": f"opt{j}", "description": "typically " +
                 ", ".join(f"{ATTR_NAMES[f]} {VALUE_WORDS[e.protos[j][f]]}" for f in range(cfg.F))}
                for j in range(M)]
        st = "Observed evidence: " + "; ".join(f"{ATTR_NAMES[f]}={VALUE_WORDS[e.values[f]]}" for f in order) + "."
        probs = {f"opt{j}": float(e.r[j]) for j in range(M)}; s = sum(probs.values())
        probs = {k: v / s for k, v in probs.items()}
        q = {"type": "choice", "instructions": "Which option best matches the observed evidence?", "options": opts}
        yield rec(f"post-{i}", f"post-{i}", "known_posterior", st, q,
                  {"kind": "categorical_distribution", "probabilities": probs,
                   "source_type": "exact_posterior", "oracle_version": "genworld-v1"},
                  {"source_id": "genworld"})


SOURCES = [  # (name, adapter, cap) — v2.1: rebalanced to JevBench's topic x question-kind shape
    ("dbpedia", dbpedia, 100000), ("multinli", multinli, 120000), ("goemotions", goemotions, 43000),
    ("clinc", clinc, 15000), ("arc", arc, None), ("boolq", boolq, None),
    ("domains", synth_domains, 500000),
    ("policies", synth_policies, 120000), ("posterior", synth_posterior, 60000),
]


def split_of(r):
    h = int(hashlib.md5(r["split_group_id"].encode()).hexdigest(), 16) / 2 ** 128
    return "train" if h < 0.92 else "dev"


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    fh = {sp: open(f"{out}/{sp}.jsonl", "w") for sp in ("train", "dev")}
    rng = np.random.default_rng(0)
    n = 0; errs = 0; by = Counter(); prov = Counter(); typ = Counter(); sp_c = Counter()
    for name, adapter, cap in SOURCES:
        c = CAP if TINY else cap
        try:
            k = 0
            for r in adapter(c, rng):
                e = schema.validate_record(r)
                if e:
                    errs += 1
                    if errs <= 5:
                        print("  VALIDATION", r["record_id"], e[:2])
                    continue
                r["metadata"]["split"] = split_of(r)
                fh[r["metadata"]["split"]].write(json.dumps(r) + "\n")
                n += 1; k += 1; by[name] += 1; prov[r["target"]["source_type"]] += 1
                typ[r["model_input"]["question"]["type"]] += 1; sp_c[r["metadata"]["split"]] += 1
            print(f"  {name:12} +{k}")
        except Exception as ex:
            print(f"  {name:12} ERR {str(ex)[:120]}")
    for f in fh.values():
        f.close()
    card = {"total": n, "validation_errors": errs, "by_source": dict(by), "by_provenance": dict(prov),
            "by_type": dict(typ), "by_split": dict(sp_c)}
    json.dump(card, open(f"{out}/data_card.json", "w"), indent=2)
    print("\n=== v2 corpus ===")
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
