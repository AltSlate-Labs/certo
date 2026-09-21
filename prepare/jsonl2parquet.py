"""Convert the canonical JSONL corpus to parquet (for HF's viewer + fast loading).

Flat, filterable columns (state, task_family, provenance, question_type, split, source_id) plus the
nested parts (question, target, metadata) as JSON strings, so heterogeneous targets/options round-trip
cleanly. Reconstruct a record with json.loads on those three columns.

    python jsonl2parquet.py <corpus_dir>
"""
import sys, json, glob, os
import pandas as pd


def flat(r):
    mi = r["model_input"]
    return {
        "record_id": r["record_id"], "state_id": r["state_id"], "task_family": r["task_family"],
        "provenance": r["target"]["source_type"], "question_type": mi["question"]["type"],
        "split": r["metadata"].get("split"), "source_id": r["metadata"].get("source_id"),
        "state": mi["state"],
        "question": json.dumps(mi["question"], ensure_ascii=False),
        "target": json.dumps(r["target"], ensure_ascii=False),
        "metadata": json.dumps(r["metadata"], ensure_ascii=False),
    }


def main():
    d = sys.argv[1]
    for path in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        sp = os.path.splitext(os.path.basename(path))[0]
        rows = [flat(json.loads(l)) for l in open(path)]
        out = os.path.join(d, f"{sp}.parquet")
        pd.DataFrame(rows).to_parquet(out, index=False)
        print(f"{sp}: {len(rows)} rows -> {out}  ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
