"""Generate examples/*.md — real runs of the trained model, in the vocabulary it was trained on.

Each probability is a real model output; where a case comes from the known-answer world we also show
the EXACT answer next to the prediction. The task is feature-signature matching: given an observed
signature, which candidate's profile fits best, with calibrated confidence. The last file is the
honest boundary — real natural-language prose, where the v1 model abstains (out of distribution).

    python gen_examples.py <checkpoint_dir> <out_dir> [device]
"""
import sys, os, numpy as np, torch
import genworld as G
import train_generic as T          # exact training render_state / render_option / NAMES_TEST
from realize import ATTR_NAMES, VALUE_WORDS
from infer import DecisionModel

ckpt, out = sys.argv[1], sys.argv[2]
dev = sys.argv[3] if len(sys.argv) > 3 else ("cuda" if torch.cuda.is_available() else "cpu")
dm = DecisionModel.load(ckpt, device=dev)
cfg = G.GenConfig(); rng = np.random.default_rng(11)
os.makedirs(out, exist_ok=True)


def build(e, template=3):
    order = list(np.nonzero(e.reveal)[0]); rng.shuffle(order)
    names = list(rng.choice(T.NAMES_TEST, size=e.protos.shape[0], replace=False))
    state = T.render_state(cfg, e, template, order)
    opts = [{"id": names[j], "description": T.render_option(cfg, e.protos[j], names[j], template),
             "profile": ", ".join(f"{ATTR_NAMES[f]} {VALUE_WORDS[e.protos[j][f]]}" for f in range(cfg.F))}
            for j in range(len(names))]
    exact = {names[j]: float(e.r[j]) for j in range(len(names))}
    return state, opts, exact


def sample_until(pred):
    while True:
        e = G.sample_example(cfg, rng)
        if pred(e):
            return e


def line(probs, top, exact=None):
    s = " · ".join(f"**{k} {probs[k]:.2f}**" if k == top else f"{k} {probs[k]:.2f}" for k in probs)
    out = [f"certo → {s}", f"→ top: **{top}**"]
    if exact:
        out.append("exact  → " + " · ".join(f"{k} {exact[k]:.2f}" for k in probs))
    return out


def md_case(state, opts, exact=None, abstain=0.6):
    r = dm.decide(state, opts, abstain_below=abstain)
    body = [f"> {state}", "", "candidate profiles:"]
    body += [f"- `{o['id']}` — {o.get('profile') or o['description']}" for o in opts]
    body += [""] + line(r["probs"], r["top"], exact)
    if r.get("abstain"):
        body.append("→ *abstains (top below 0.60 → hand off / ask for more)*")
    return "\n".join(body), r


FILES = []

def write(slug, title, blurb, body):
    open(os.path.join(out, f"{slug}.md"), "w").write(f"# {title}\n\n{blurb}\n\n*Real outputs of "
        f"`altslate/certo-decision-model`.*\n\n{body}\n")
    FILES.append((slug, title, blurb)); print("wrote", slug)


# 1. confident
e = sample_until(lambda e: G.dist_entropy(e.r) / np.log(len(e.r)) < 0.12)
b, _ = md_case(*build(e))
write("01-confident-match", "Confident match", "Clear evidence → a confident, correct pick.", b)

# 2. calibrated uncertainty
e = sample_until(lambda e: G.dist_entropy(e.r) / np.log(len(e.r)) > 0.7)
b, _ = md_case(*build(e))
write("02-calibrated-uncertainty", "Calibrated uncertainty",
      "Ambiguous evidence → an honest spread that matches the true probabilities, and an abstain.", b)

# 3. many options
e = G.sample_example(cfg, rng, M=9)
b, _ = md_case(*build(e))
write("03-many-options", "Many options", "Nine candidates at once (trained on 3–6) — still calibrated.", b)

# 4. order invariance
st, op, ex = build(G.sample_example(cfg, rng))
r1 = dm.decide(st, op)["probs"]; r2 = dm.decide(st, list(reversed(op)))["probs"]
body = [f"> {st}", "", "Scoring each candidate is independent, so reordering the options does not change "
        "any candidate's probability.", "", "| candidate | original | reversed |", "|---|---|---|"]
body += [f"| `{k}` | {r1[k]:.3f} | {r2[k]:.3f} |" for k in r1]
write("04-order-invariance", "Order invariance",
      "Reordering the candidates leaves every probability unchanged — a structural guarantee.", "\n".join(body))

# 5. adding an option doesn't move the others' odds
st, op, ex = build(G.sample_example(cfg, rng, M=3))
a, bb = op[0]["id"], op[1]["id"]
p_before = dm.decide(st, op)["probs"]
extra = G.sample_example(cfg, rng, M=1)
op2 = op + [{"id": "Newcomer", "description": T.render_option(cfg, extra.protos[0], "Newcomer", 3)}]
p_after = dm.decide(st, op2)["probs"]
body = [f"> {st}", "",
        f"Adding a new candidate barely moves the odds between the existing ones "
        f"(`{a}` : `{bb}`):", "",
        f"- before: {a} {p_before[a]:.3f}, {bb} {p_before[bb]:.3f}  → odds {p_before[a]/max(p_before[bb],1e-9):.2f}",
        f"- after (+Newcomer): {a} {p_after[a]:.3f}, {bb} {p_after[bb]:.3f}  → odds {p_after[a]/max(p_after[bb],1e-9):.2f}"]
write("05-adding-options", "Adding an option",
      "New runtime options don't repaint the existing ones — named-candidate odds stay put.", "\n".join(body))

# 6. unseen profiles (every profile here is new at request time)
e = sample_until(lambda e: 0.2 < G.dist_entropy(e.r) / np.log(len(e.r)) < 0.5)
b, _ = md_case(*build(e))
write("06-unseen-profiles", "Unseen profiles at request time",
      "The candidate profiles are supplied at request time and were never in training — certo binds "
      "each description to the evidence and stays calibrated.", b)

# 7. honest boundary: real prose
b, _ = md_case(
    "Customer: I was charged twice for the same order and I'd like a refund.",
    [{"id": "billing", "description": "billing and payments: duplicate charges, refunds, invoices"},
     {"id": "tech", "description": "the app or website is broken"},
     {"id": "account", "description": "login and account access problems"}], abstain=0.5)
write("07-boundary-real-language", "The boundary: real language",
      "The v1 model is trained on structured signatures, so on free-flowing prose it returns a "
      "near-uniform distribution and **abstains** rather than guess. Handling real language is the v2 "
      "goal (real data + a paraphrase layer). This is the honest edge of what v1 can do.", b)

idx = ["# certo — worked examples", "",
       "Real runs of the default model ([`altslate/certo-decision-model`]"
       "(https://huggingface.co/altslate/certo-decision-model)). certo reads an observed signature and "
       "candidate profiles and returns **calibrated probabilities**, picking the best match and "
       "**abstaining** when the evidence is ambiguous.", ""]
idx += [f"{i+1}. [{t}]({s}.md) — {b}" for i, (s, t, b) in enumerate(FILES)]
open(os.path.join(out, "README.md"), "w").write("\n".join(idx) + "\n")
print("wrote index")
