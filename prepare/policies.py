"""Executable-policy generator: varied decision BOUNDARIES with an independent evaluator.

Each family samples a rule + facts (including boundary and counterfactual cases), computes the
target by executing the rule (not by an LLM), and renders the observable facts + rule into text.
Targets are deterministic -> source_type "program_verified" (a one-hot / hard target is correct
here; this is a *known* rule outcome, not an argmax of a genuinely uncertain posterior).
"""
from __future__ import annotations
import numpy as np
from schema import record

NAMES = ["Ava", "Ben", "Cy", "Dee", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jo"]


def _rec(idx, fam, state, question, target, meta):
    sid = f"{fam}-{idx:06d}"
    return record(f"{sid}-q0", sid, fam, {"state": state, "question": question}, target,
                  {"split": None, "source_id": "own_policy_generator", "source_revision": "v1", **meta})


# --------------------------------------------------------------------------- choice: threshold
def threshold_return(rng, idx):
    limit = int(rng.choice([7, 14, 30]))
    days = limit + int(rng.integers(-3, 4))            # cluster around the boundary
    unused = rng.random() < 0.8
    days_known = rng.random() < 0.85
    if not days_known:      ans = "c"
    elif unused and days <= limit: ans = "a"
    else:                   ans = "b"
    state = f"The item is {'unused' if unused else 'used'}. " + (
        f"The return was requested {days} days after purchase." if days_known
        else "The customer has not said when the return was requested.")
    q = {"type": "choice",
         "instructions": "Select the action required by the supplied return policy.",
         "rubric": f"Accept an unused item if the return was requested within {limit} days (inclusive). "
                   f"Decline when a stated condition fails. Request information only when eligibility "
                   f"cannot be determined from the stated facts.",
         "options": [{"id": "a", "description": "Accept the return under the policy."},
                     {"id": "b", "description": "Decline the return under the policy."},
                     {"id": "c", "description": "Request information needed to determine eligibility."}]}
    return _rec(idx, "supplied_policy_threshold", state, q,
                {"kind": "categorical_label", "label_id": ans, "source_type": "program_verified"},
                {"domain": "returns", "rule_family": "threshold", "rule_signature": "unused_AND_days_LE_limit"})


# --------------------------------------------------------------------------- choice: conjunction
def conjunction_approval(rng, idx):
    min_income = int(rng.choice([40, 60, 80]))
    income = min_income + int(rng.integers(-15, 16))    # near the threshold, in $k
    credit_ok = rng.random() < 0.6
    if income < min_income:   ans = "b"                 # income fails
    elif credit_ok:           ans = "a"                 # both pass
    else:                     ans = "c"                 # income ok, credit not
    state = f"Applicant income: ${income}k. Credit check: {'clear' if credit_ok else 'not clear'}."
    q = {"type": "choice",
         "instructions": "Choose the decision required by the lending policy.",
         "rubric": f"Approve if income is at least ${min_income}k AND the credit check is clear. "
                   f"Reject if income is below ${min_income}k. Otherwise refer for manual review.",
         "options": [{"id": "a", "description": "Approve the application."},
                     {"id": "b", "description": "Reject the application."},
                     {"id": "c", "description": "Refer for manual review."}]}
    return _rec(idx, "supplied_policy_conjunction", state, q,
                {"kind": "categorical_label", "label_id": ans, "source_type": "program_verified"},
                {"domain": "lending", "rule_family": "conjunction", "rule_signature": "income_GE_min_AND_credit_ok"})


# --------------------------------------------------------------------------- score: rubric
def rubric_severity(rng, idx):
    factors = ["repeat report", "safety keyword", "vulnerable user", "prior escalation", "financial loss"]
    present = rng.random(len(factors)) < 0.4
    score = int(min(present.sum(), 3))
    listed = [f for f, p in zip(factors, present) if p]
    state = "Signals present: " + (", ".join(listed) if listed else "none") + "."
    q = {"type": "score", "levels": 4,
         "instructions": "Assign the priority level under the rubric.",
         "rubric": "Priority level = number of listed risk signals present, capped at 3 (levels 0-3)."}
    probs = [0.0] * 4; probs[score] = 1.0
    return _rec(idx, "supplied_policy_rubric_score", state, q,
                {"kind": "score_distribution", "probabilities": probs, "level": score,
                 "source_type": "program_verified"},
                {"domain": "triage", "rule_family": "rubric_score", "rule_signature": "count_signals_cap3"})


# --------------------------------------------------------------------------- independent-binary: panel
def expert_panel(rng, idx):
    m = int(rng.integers(3, 6))
    names = list(rng.choice(NAMES, size=m, replace=False))
    difficulty = float(rng.uniform(0.2, 0.8))
    skills = {n: float(rng.uniform(0.0, 1.0)) for n in names}
    state = f"Task difficulty D = {difficulty:.2f}. " + " ".join(
        f"Expert {n}: skill {skills[n]:.2f}." for n in names)
    q = {"type": "independent_binary",
         "instructions": "For each expert, decide whether they meet the quality bar for this task.",
         "rubric": "An expert meets the bar when their skill is at least the task difficulty D.",
         "options": [{"id": n, "description": f"Expert {n}"} for n in names]}
    values = {n: int(skills[n] >= difficulty) for n in names}
    return _rec(idx, "supplied_policy_panel", state, q,
                {"kind": "bernoulli_labels", "values": values,
                 "observed_mask": {n: True for n in names}, "source_type": "program_verified"},
                {"domain": "routing", "rule_family": "multi_condition", "rule_signature": "skill_GE_difficulty"})


# --------------------------------------------------------------------------- noul (single yes/no)
def qualifies_binary(rng, idx):
    a = rng.random() < 0.5; b = rng.random() < 0.5; ok = a and b
    state = f"Condition A is {'met' if a else 'not met'}. Condition B is {'met' if b else 'not met'}."
    q = {"type": "binary",
         "instructions": "Does the case qualify? It qualifies only if both A and B are met.",
         "options": [{"id": "yes", "description": "the case qualifies"},
                     {"id": "no", "description": "the case does not qualify"}]}
    return _rec(idx, "supplied_policy_binary", state, q,
                {"kind": "categorical_label", "label_id": "yes" if ok else "no",
                 "source_type": "program_verified"},
                {"domain": "eligibility", "rule_family": "binary_conjunction", "rule_signature": "A_AND_B"})


FAMILIES = [threshold_return, conjunction_approval, rubric_severity, expert_panel, qualifies_binary]


def generate(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        out.append(FAMILIES[i % len(FAMILIES)](rng, i))
    return out
