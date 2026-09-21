# certo — worked examples

Real runs of the default model ([`altslate/certo-decision-model`](https://huggingface.co/altslate/certo-decision-model)). certo reads an observed signature and candidate profiles and returns **calibrated probabilities**, picking the best match and **abstaining** when the evidence is ambiguous.

1. [Confident match](01-confident-match.md) — Clear evidence → a confident, correct pick.
2. [Calibrated uncertainty](02-calibrated-uncertainty.md) — Ambiguous evidence → an honest spread that matches the true probabilities, and an abstain.
3. [Many options](03-many-options.md) — Nine candidates at once (trained on 3–6) — still calibrated.
4. [Order invariance](04-order-invariance.md) — Reordering the candidates leaves every probability unchanged — a structural guarantee.
5. [Adding an option](05-adding-options.md) — New runtime options don't repaint the existing ones — named-candidate odds stay put.
6. [Unseen profiles at request time](06-unseen-profiles.md) — The candidate profiles are supplied at request time and were never in training — certo binds each description to the evidence and stays calibrated.
7. [The boundary: real language](07-boundary-real-language.md) — The v1 model is trained on structured signatures, so on free-flowing prose it returns a near-uniform distribution and **abstains** rather than guess. Handling real language is the v2 goal (real data + a paraphrase layer). This is the honest edge of what v1 can do.
