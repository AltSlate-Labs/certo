"""Canonical decision-record schema + validator (decision-data-v1).

One question per record. Only `model_input` is ever shown to the model; everything else is target /
metadata / audit. Targets carry explicit provenance (program_verified | exact_posterior |
human_annotation | teacher_estimate). See notes/general-decision-model-data-preparation.md.
"""
from __future__ import annotations

SCHEMA_VERSION = "decision-data-v1"
QUESTION_TYPES = {"choice", "binary", "independent_binary", "score"}
TARGET_KINDS = {"categorical_label", "categorical_distribution", "binary_label",
                "bernoulli_labels", "score_distribution"}
MODEL_INPUT_KEYS = {"state", "question"}
QUESTION_KEYS = {"type", "instructions", "rubric", "options", "subset", "class_to_level", "levels"}


def record(record_id, state_id, task_family, model_input, target, metadata, split_group_id=None):
    return {
        "schema_version": SCHEMA_VERSION, "record_id": record_id, "state_id": state_id,
        "split_group_id": split_group_id or state_id, "task_family": task_family,
        "model_input": model_input, "target": target, "metadata": metadata,
    }


def _opt_ids(mi):
    return [o["id"] for o in mi["question"].get("options", [])]


def validate_record(r) -> list[str]:
    """Return a list of problems ([] == valid). Enforces the input-field boundary + target sanity."""
    e = []
    for k in ("schema_version", "record_id", "state_id", "split_group_id", "task_family",
              "model_input", "target", "metadata"):
        if k not in r:
            e.append(f"missing top-level field: {k}")
    if e:
        return e
    mi, q, t = r["model_input"], r["model_input"].get("question", {}), r["target"]

    # input-field boundary: model_input holds ONLY state + question; question only allowed keys
    extra = set(mi) - MODEL_INPUT_KEYS
    if extra:
        e.append(f"model_input has non-allowlisted keys (leak risk): {sorted(extra)}")
    if "state" not in mi or not isinstance(mi["state"], str):
        e.append("model_input.state missing or not a string")
    qextra = set(q) - QUESTION_KEYS
    if qextra:
        e.append(f"question has non-allowlisted keys: {sorted(qextra)}")
    if q.get("type") not in QUESTION_TYPES:
        e.append(f"bad question.type: {q.get('type')}")
    # no obvious answer leak in the visible text
    for field in ("state", "instructions", "rubric"):
        txt = (mi.get(field) or q.get(field) or "")
        if isinstance(txt, str) and ("label_id" in txt or "correct_answer" in txt):
            e.append(f"possible answer leak in {field}")

    kind = t.get("kind")
    if kind not in TARGET_KINDS:
        e.append(f"bad target.kind: {kind}")
    if "source_type" not in t:
        e.append("target.source_type missing (provenance required)")

    if kind == "categorical_label":
        if t.get("label_id") not in _opt_ids(mi):
            e.append("categorical_label.label_id not among options")
    elif kind == "categorical_distribution":
        p = t.get("probabilities", {})
        ids = set(_opt_ids(mi))
        if set(p) - ids:
            e.append("distribution keys not a subset of option ids")
        if any(v < -1e-9 for v in p.values()):
            e.append("negative probability")
        if abs(sum(p.values()) - 1.0) > 1e-6:
            e.append(f"distribution does not sum to 1 ({sum(p.values()):.6f})")
    elif kind == "binary_label":
        if t.get("value") not in (0, 1):
            e.append("binary_label.value must be 0/1")
    elif kind == "bernoulli_labels":
        vals, mask = t.get("values", {}), t.get("observed_mask", {})
        if set(vals) != set(mask):
            e.append("bernoulli values/observed_mask key mismatch")
        for k, v in vals.items():
            if v not in (0, 1, None):
                e.append(f"bernoulli value for {k} not in 0/1/null")
            if v is None and mask.get(k) is not False:
                e.append(f"unobserved event {k} must have observed_mask False")
    elif kind == "score_distribution":
        p = t.get("probabilities", [])
        if abs(sum(p) - 1.0) > 1e-6 or any(v < -1e-9 for v in p):
            e.append("score distribution invalid")
    return e
