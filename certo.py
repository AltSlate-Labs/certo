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
"""
from infer import DecisionModel  # noqa: F401

__all__ = ["DecisionModel"]
__version__ = "0.1.0"
