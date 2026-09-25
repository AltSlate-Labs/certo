"""certo — small models that know how sure they are.

Calibrated, non-generative decision models. Quick start:

    from huggingface_hub import snapshot_download
    from certo import DecisionModel

    m = DecisionModel.load(snapshot_download("altslate/certo-decision-model"))
    r = m.decide(
        state="We measured tempo as gale, texture as dawn, density as ember.",
        options=[{"id": "Loam", "description": "typically texture dawn, tempo gale, density ember"},
                 {"id": "Dune", "description": "typically texture gale, tempo dawn, density dawn"},
                 {"id": "Moor", "description": "typically texture ember, tempo frost, density frost"}],
        abstain_below=0.6)
    r["probs"]    # {'Loam': 0.99, 'Dune': 0.0, 'Moor': 0.0}  — calibrated
    r["top"]      # 'Loam'
    r["abstain"]  # False

Generative reasoner (e.g. Certo-R1) behind the same API:

    from certo import ReasoningDecisionModel
    m = ReasoningDecisionModel.load("altslate/certo-r1-qwen3-4b", device="cuda")
    r = m.decide("A parcel weighs 2.4 kg; shipping is $3/kg rounded up; charged $9.",
                 [{"id": "yes", "description": "the amount charged is correct"},
                  {"id": "no",  "description": "the amount charged is wrong"}],
                 instructions="Was the customer charged correctly?")
    r["top"]        # "yes"
    r["rationale"]  # brief reasoning ending in FINAL ANSWER
"""
from infer import DecisionModel  # noqa: F401
from infer_reasoner import ReasoningDecisionModel  # noqa: F401

__all__ = ["DecisionModel", "ReasoningDecisionModel"]
__version__ = "0.2.0"
