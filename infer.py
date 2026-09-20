"""certo inference: load a trained decision model and make a decision locally.

    from infer import DecisionModel
    m = DecisionModel.load("path/to/checkpoint")
    r = m.decide("charged twice, want a refund",
                 [{"id": "billing", "description": "charges and payments"},
                  {"id": "tech",    "description": "app problems"}])
    r["probs"]    # {"billing": 0.86, "tech": 0.14}  -- calibrated
    r["top"]      # "billing"
    r["abstain"]  # True if the top probability is below the threshold you set

Non-generative: one forward pass -> typed probabilities, no text to parse. The model scores each
option from its description against the state (runtime options; order-invariant by construction).
"""
from __future__ import annotations
import os, json, torch
from transformers import AutoTokenizer
from model_generic import GenericDecisionModel


class DecisionModel:
    def __init__(self, model, tokenizer, cfg):
        self.model = model.eval()
        self.tok = tokenizer
        self.cfg = cfg

    @classmethod
    def load(cls, path, device="cpu"):
        cfg = json.load(open(os.path.join(path, "certo_config.json")))
        model = GenericDecisionModel(cfg["backbone"], heads=cfg.get("heads", 8))
        model.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location=device))
        model.to(device)
        tok = AutoTokenizer.from_pretrained(path)
        return cls(model, tok, {**cfg, "device": device})

    @torch.no_grad()
    def decide(self, state, options, max_state_len=64, max_option_len=48, abstain_below=None):
        """state: str. options: list of {"id"/"name", "description"}. Returns calibrated probs."""
        dev = self.cfg["device"]
        se = self.tok([state], padding="max_length", truncation=True,
                      max_length=max_state_len, return_tensors="pt")
        oe = self.tok([o["description"] for o in options], padding="max_length", truncation=True,
                      max_length=max_option_len, return_tensors="pt")
        M = len(options)
        logit = self.model(se["input_ids"].to(dev), se["attention_mask"].to(dev),
                           oe["input_ids"].unsqueeze(0).to(dev), oe["attention_mask"].unsqueeze(0).to(dev),
                           torch.ones(1, M, dtype=torch.bool, device=dev))
        p = torch.softmax(logit, 1)[0].cpu().tolist()
        ids = [o.get("id", o.get("name", str(i))) for i, o in enumerate(options)]
        probs = {ids[i]: round(p[i], 4) for i in range(M)}
        res = {"probs": probs, "top": max(probs, key=probs.get)}
        if abstain_below is not None:
            res["abstain"] = max(probs.values()) < abstain_below
        return res
