"""certo reasoning inference: run a *generative* Certo decision model (e.g. Certo-R1) behind the same API as
the non-generative DecisionModel.

    from infer_reasoner import ReasoningDecisionModel
    m = ReasoningDecisionModel.load("altslate/certo-r1-qwen3-4b", device="cuda")
    r = m.decide("A shift ends 20:00 UTC; next begins 06:00 CET (UTC+1) next day; >=11h rest required.",
                 [{"id": "yes", "description": "at least 11 hours of real rest"},
                  {"id": "no",  "description": "less than 11 hours of real rest"}],
                 instructions="Is the minimum rest period satisfied?")
    r["top"]        # "yes"
    r["rationale"]  # the model's one/two-sentence reasoning
    r["probs"]      # {"yes": 1.0, "no": 0.0}  -- one-hot (a reasoner emits an answer, not a calibrated dist)

Unlike DecisionModel (encoder, calibrated probabilities in one forward pass), this reasons briefly and returns
the chosen option. Same option format ({"id"/"name", "description"}); the answer is mapped back to your ids.
"""
from __future__ import annotations
import re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_LET = [chr(65 + i) for i in range(26)]
_CONCISE = "Reason in at most two short sentences, then end with a line 'FINAL ANSWER: <letter>'."


class ReasoningDecisionModel:
    def __init__(self, model, tokenizer, device):
        self.model = model.eval()
        self.tok = tokenizer
        self.device = device

    @classmethod
    def load(cls, path, device="cuda", dtype=torch.bfloat16):
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, device_map={"": 0} if "cuda" in device else None)
        if "cuda" not in device:
            model.to(device)
        return cls(model, tok, device)

    def _prompt(self, state, options, instructions):
        opts = [(_LET[i], o["description"]) for i, o in enumerate(options)]
        head = (instructions + "\n\n") if instructions else ""
        body = (f"{head}Context: {state}\n\nOptions:\n"
                + "\n".join(f"{L}) {d}" for L, d in opts) + f"\n\n{_CONCISE}")
        return self.tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)

    @torch.no_grad()
    def decide(self, state, options, instructions=None, max_new_tokens=96):
        """state: str. options: list of {"id"/"name", "description"}. Returns {"top","answer","rationale","probs"}."""
        ids = [o.get("id", o.get("name", str(i))) for i, o in enumerate(options)]
        letters = [_LET[i] for i in range(len(options))]
        prompt = self._prompt(state, options, instructions)
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        g = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        up = text.upper(); pick = None
        for pat in (r"FINAL ANSWER\D{0,15}\b([A-Z])\b", r"\b([A-Z])\)", r"\b([A-Z])\b"):
            m = [x for x in re.findall(pat, up) if x in letters]
            if m:
                pick = m[-1]; break
        idx = letters.index(pick) if pick in letters else 0
        top = ids[idx]
        probs = {i: (1.0 if i == top else 0.0) for i in ids}
        return {"top": top, "answer": top, "rationale": text.strip(), "probs": probs,
                "gen_tokens": int(g.shape[1] - enc["input_ids"].shape[1])}
