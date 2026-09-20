# JEV — a small calibrated decision model

*Design doc. Branch: `jev`. Status: v0 built + measured — see §9. Code: `datagen.py`, `model.py`, `train.py`.*

## 0. What this is

A **non-generative decision model**: given a shared *state* and one or more *questions*,
it emits a **calibrated probability distribution** over runtime-supplied answer options —
no text generation, no reasoning trace, just typed probabilities.

This reconstructs the interesting core of "Jev" (the TypeSafe decision model). The
architecture is 2021-era territory (Perceiver-IO / GLiClass query-a-shared-encoding). The
part that is actually hard — and the part we build the whole project around — is
**calibration**: emitting honest probabilities, learned from **100% synthetic data** whose
ground-truth answer distribution is *analytically known*.

Two deliverables, in order:
1. **Synthetic data generator** (build first) — the moat.
2. **Model architecture** — the easy 20%, specified here, built after the data is trusted.

### The organizing principle

We generate examples from a latent probabilistic world where `P(answer | state)` is
computable in **closed form**. Everything good follows from that one decision:

- Train against the **true posterior** `r`, not a hard label (proper scoring rule).
- Measure the model against the **Bayes-optimal** distribution — **posterior fidelity**
  (`KL(r‖p)`, `TV`), a *stronger* test than aggregate calibration (`ECE`, `Brier`), and
  neither is available on real data where the exact target is unknown.
- A single knob controls target **entropy**, so calibration is tested across the whole
  confidence spectrum, not just easy peaked cases.
- The two falsifiable experiments from the reconstruction (option-invariance,
  question-isolation) become *properties of the data*, so they can be checked, not asserted.

---

## 1. The synthetic world

A **conditional naive-Bayes evidence world**. Minimal world with genuine, *tunable*
aleatoric uncertainty and an exact posterior.

### Generative model

- `K` classes (the answer universe).
- **Likelihood tables `θ_f(· | y)` are fixed across the entire corpus** (drawn once at
  corpus-build time, then frozen). This is load-bearing: if the tables varied per world and
  were *hidden*, the world-specific posterior would be **unrecoverable** from the state — the
  model would have no way to know which likelihoods to apply. Fixed tables let it learn them
  from the training distribution. (v1: expose the tables in-context, or give in-context
  examples to infer them from — only then may they vary per world.)
- **Prior `π ~ Dirichlet(α)` is drawn per world and written into the state** (see §3), so the
  model reads the base rates instead of memorizing one prior. Keep a **fixed-prior** corpus as
  a debugging baseline.
- Draw the true class `y ~ π`.
- `F` feature slots. Slot `f` has likelihood `θ_f(· | y)` over `V_f` values.
  **Discriminativeness** varies per slot (a concentration/temperature) but is fixed across
  worlds: near-uniform slots are weak/distractor evidence; peaked slots are strong.
- **Reveal**: each slot is revealed with probability `ρ`. A revealed slot emits
  `v_f ~ θ_f(· | y)`; unrevealed slots contribute nothing. `ρ` is the master knob on
  *average* evidence quantity → posterior entropy (per-example entropy can move either way — a
  single conflicting item can *raise* it; see §5).
- Optional **noise items**: slots with identical likelihood across all classes (pure
  distractors) to test robustness to irrelevant evidence.

The **state** `E` = the multiset of revealed `(feature, value)` pairs, order randomized.

### Exact posterior (the target)

By conditional independence given `y`:

$$P(y=k \mid E)\;\propto\;\pi_k \prod_{f \in \text{revealed}} \theta_f(v_f \mid k),\qquad
\text{normalized over } k=1..K.$$

This is the oracle target `r`. Add/remove an evidence item ⇒ multiply/divide one factor ⇒
recompute trivially. That tractability is what makes the entropy sweep and question
construction cheap.

### Why this task is right-sized

The model never sees the likelihood tables. From serialized tokens it must (i) parse which
`(feature,value)` pairs are present, (ii) learn each pair's per-class evidence weight
`log θ_f(v|k)`, (iii) sum with the log-prior, (iv) softmax. This is **log-linear evidence
integration**:

- getting the **sign** of the log-likelihood-ratios right ⇒ **accuracy**;
- getting their **magnitude** right ⇒ **calibration**.

The task cleanly separates the two — exactly the phenomenon we care about — while keeping an
exact target. Not trivial (magnitudes must be learned), not unbounded (closed-form oracle).

### v1 extension — beyond naive-Bayes

Add a world variant with **feature interactions** (XOR-like: a feature's evidence value only
matters given another's), so the posterior is *not* factorizable and the model must combine
features nonlinearly. Keep it closed-form by enumerating a small explicit latent. Noted here,
not built in v0.

---

## 2. Mapping to the question interface

Same state `E`, multiple **independent** questions, each with an exact target. Because every
question's target is computed independently from the same `E`, the dataset *embodies
question-isolation* — no question can contaminate another.

**Choice (categorical).** Options = a runtime subset/relabeling `S ⊆ {1..K}`. When `S ⊂ {1..K}`
(some classes omitted) the option set includes an explicit **`OUT`** ("the answer lies outside
these options"). The target over `S ∪ {OUT}` uses the **raw posterior** — *not* a
renormalization:

$$r(k)=P(y=k\mid E)\ (k\in S),\qquad
r_{\text{OUT}}=\sum_{k\notin S}P(y=k\mid E)=1-\sum_{k\in S}P(y=k\mid E).$$

Two consequences, both intentional:
- **Odds between named classes are invariant** to the option set:
  `r(a)/r(b) = P(a|E)/P(b|E)` for `a,b ∈ S`, independent of `S`. This is the property the
  option-independent read must preserve — **test it on named classes only.**
- **`r_OUT` is *not* invariant** — it is the complement mass, so it *must* grow as more classes
  are omitted. A fixed, independently-scored `OUT` option cannot produce this. **Test `OUT`
  separately.**

`OUT` is **not** uncertainty abstention: it asserts the answer is outside the listed options,
and a model can predict it *confidently*. "I'm unsure, request review" is a different decision,
not modeled here. (If `S = {1..K}`, no `OUT` — the target is the full posterior. For
label-based eval without `OUT`, condition on samples with `y ∈ S`.)

*Option binding:* each option names its class by a **stable class id** (evidence semantics are
fixed corpus-wide, so the id is the thing the model learns to bind to). The **presented subset
and their order are randomized per example**, so the model must bind id→evidence and stay
invariant to position and to which other classes appear — not memorize a slot. (Unseen class
ids are the deferred generalization test.)

**Binary / Noul (yes-no).** Predicate = random subset `B ⊂ {1..K}` ("is it one of …?").

$$p=\sum_{k\in B}P(y=k\mid E).$$

Spans `[0,1]` fully as `B`/evidence vary — good coverage for the sigmoid head.

**Score (ordinal).** A runtime **class→level map** (stated in the question, like options) bins
classes into `L` levels. Target = posterior mass per level — a full distribution, so ordinary
**softmax over levels** supports it; scalar = `E[level]`. Ordinal/monotonic structure is a
later ablation, not v0.

---

## 3. Example schema

JSONL (human-readable) + a packed tensor form for training. Questions sharing a `world_id`
share one state.

```jsonc
{
  "world_id": 12034, "seed": 12034,
  "params": { "K": 8, "F": 12, "rho": 0.5, "prior_alpha": 1.0 },
  "prior":      [0.20,0.05,0.15,0.10,0.08,0.22,0.12,0.08],          // per-world, stated in input
  "state":      [ {"f": 3, "v": 7}, {"f": 9, "v": 1}, ... ],        // revealed multiset
  "state_text": "prior: c0=.20 c1=.05 ... | f3=v7 | f9=v1 | ...",   // prior IS part of the input
  "y_true": 5,                                                      // accuracy + self-check only
  "questions": [
    { "type": "choice",                                            // named = raw P(k|E); OUT = Σ_{k∉S} P(k|E)
      "options": [ {"tok": "A", "class": 5}, {"tok": "B", "class": 2}, {"tok": "OUT", "class": null} ],
      "target": [0.61, 0.16, 0.23], "meta": { "entropy": 1.03 } },
    { "type": "binary", "subset": [5, 2],
      "target": 0.77, "meta": { "entropy": 0.78 } },
    { "type": "score", "levels": 3, "class_to_level": [0,0,1,1,2,2,2,2],
      "target": [0.28, 0.30, 0.42], "scalar": 1.14 }
  ]
}
```

The **exact target is always stored**; training uses it (soft loss). `y_true` is kept only
for accuracy and the reliability self-check.

---

## 4. Surface realization — symbolic vs text

Same symbolic core, pluggable realizer. The symbolic core owns all probability math;
realizers only change the surface string.

- **Symbolic mode (v0).** Abstract tokens (`f3=v7`, `opt=A`). Tiny custom vocab, tiny
  from-scratch encoder, runs on a Mac/CPU. Isolates the *calibration* question from the *NLP*
  question. This is where we prove the objective and the architecture.
- **Templated-text mode (v1).** A light grammar renders each `(feature,value)` into a short
  clause ("Account age: 3 days.") and options into label phrases, enabling a pretrained
  encoder (ModernBERT). **Targets are identical** — only the surface changes — so we cleanly
  separate "can it calibrate" from "can it read."

---

## 5. Verification built into the generator

We trust no model number until the generator verifies its own math. All cheap, all automated.

1. **Reliability of oracle labels (with tolerance).** Bin by predicted `r_k`; empirical
   `y=k` frequency lies on the diagonal **in expectation** — finite samples fluctuate, so
   check each bin against a **binomial confidence interval**, not equality. Given correct
   Bayes math, `E[𝟙(y=k) | r_k] = r_k` is a theorem; a curve that leaves the CIs means the
   posterior code is wrong. Complement with **tiny-world exhaustive checks** (small `K,F,V`:
   enumerate every evidence config and compare the closed form to brute-force enumeration
   exactly).
2. **Normalization & invariance unit tests.** Targets sum to 1 (incl. the `OUT` mass); choice
   odds invariant to added options (within tolerance); binary target = sum of the matching
   choice masses; score scalar `= Σ level·p`.
3. **Entropy coverage (empirical, not assumed).** Sweep `ρ` and *measure* the target-entropy
   distribution — higher `ρ` lowers uncertainty **on average**, but an individual revealed
   item can *raise* entropy (conflicting evidence). Confirm we actually cover low→high entropy.
4. **Monte-Carlo spot check.** For a fixed evidence config, rejection-sample `y` and confirm
   the empirical posterior matches the closed form within CI.

---

## 6. Model architecture

One hypothesis (the reconstruction's two candidates collapse: once state tokens may not read
question tokens, the state encoding is question-independent — which *is* the shared-encoder +
query-readout design). **Encode state once; query independently; typed calibrated readouts.**

- **State encoder `E_θ` (bidirectional).** Serialized evidence → `H_x ∈ ℝ^{n×d}`.
  Bidirectional, not causal: the state is a random-order multiset of evidence to *understand*,
  nothing to generate. Small: `d≈256`, 4–6 layers, symbolic vocab. **Question-agnostic** —
  never sees the questions ⇒ `H_x` is cached once, and **question-isolation and
  question-invariance are structural guarantees, not learned.**
- **Query module `Q_φ` (per question).** Encode the question **type + rubric only** — for
  Choice, *not* the option list — into query tokens; cross-attend into `H_x` (2–4 cross-attn +
  FFN layers) → an **option-independent** query read. This is the architectural condition for
  odds-invariance: if the read saw the full option set, independent final scoring would *not*
  guarantee invariant odds. Each option is then scored against this shared read.
- **Typed heads.**
  - *Choice:* per-option score `z_k = s(read, enc(option_k))`, `read` independent of the option
    set ⇒ softmax over named options gives `p(a)/p(b) = exp(z_a − z_b)`, invariant — matching
    the data. **`OUT` needs special handling:** its target is the *complement* mass, which a
    fixed independently-scored logit cannot produce (it depends on which classes are omitted).
    Model-phase options: calibrate named logits as absolute log-posteriors so
    `OUT = log(1 − Σ softmax(named))`, or marginalize an internal full-class head. Deferred to
    the model phase; datagen just emits the exact complement.
  - *Binary:* single logit → sigmoid.
  - *Score:* logits over `L` levels → softmax; scalar `= Σ level·p` (or a monotonic ordinal head).

### Objective — the actual experiment

**Three training arms, identical architecture, differing only in the target:**

1. **Soft posterior `r`** — the proper scoring rule
   `L = −Σ_k r_k log p_k` (categorical) / `BCE(p, p_true)` (binary). Minimized at `p = r`;
   full distributional supervision.
2. **Sampled true label `y ~ r`** — ordinary CE on a fresh label sampled from the posterior
   each epoch. **This also targets `r` in expectation** — it is the honest control.
3. **Argmax label `argmax(r)`** — ordinary CE on the hard label; erases uncertainty.

Arm (2) is what makes the experiment mean anything. Without it, a gap between (1) and (3) could
prove only the trivial fact that *information erased by hardening cannot be recovered*. Arm (2)
shares (3)'s "one label per example" format but keeps the uncertainty, so **(2) vs (3)**
isolates the effect of *erasing* the distribution from the effect of *soft vs hard*
supervision, and **(1) vs (2)** isolates the value of dense soft targets over sampled ones.

**Headline, narrowed to what this actually shows:** *hardening posterior targets harms
posterior fidelity.* What it does **not** establish: RLCD's benefit, or anything about Jev's
real (undisclosed) training algorithm — these are supervised objectives, not RL. **Matched
accuracy across arms is a hypothesis to test, not a claim.**

### Other designed ablations

- **Pooled-linear baseline.** Linear head on a pooled embedding (no cross-attention, no
  runtime options) — shows the query-read earns its keep.
- **Joint-option head.** Options cross-attend to each other → violates odds-invariance. The
  arm the real-Jev "add an irrelevant option, do A:B odds move?" API probe would distinguish.

### Metrics (against the analytic oracle — the luxury this design buys)

- **Posterior fidelity:** `KL(r‖p)` and `TV` to the exact posterior — a *stronger* test than
  aggregate calibration (impossible on real data, where the target is unknown).
- **Calibration:** `ECE`, adaptive-`ECE`, `Brier`, reliability curves.
- **Discrimination:** accuracy, AUROC. Sharpness-vs-calibration decomposition.
- **Invariance:** option-permutation, option-addition (odds drift), question-isolation
  (`A` alone vs `A | B,C` — invariant *by construction* of the option-independent read;
  checked **numerically within tolerance**, not bit-identical, since batching perturbs floats).
- **Generalization:** train on some `(π, ρ, discriminativeness)` regimes, test on unseen
  entropy regimes, unseen option relabelings, more classes than seen.

---

## 7. Scope discipline

- **v0 (now):** this doc → symbolic naive-Bayes world + generator + self-verification. Model
  after the data is trusted.
- **v1:** templated-text realizer (several equivalent templates, shuffled evidence order,
  held-out paraphrases — identical underlying info & targets, so language is the *only* added
  difficulty) + ModernBERT encoder; feature-interaction (non-naive-Bayes) world.
- **Later:** joint-option ablation; a genuine **RL arm** (to actually probe RLCD, distinct
  from the three *supervised* arms here); real-Jev API probe harness (the reconstruction's
  controlled experiments). *(Abstention via the `OUT` option is v0, not later.)*

**Explicitly not now:** serving kernels / the `n² + m(nr+r²)` efficiency win (a serving
concern, orthogonal to the research question), multi-hop reasoning, anything generative.

---

## 8. Decisions & open questions

**Resolved (this round):**
- **Prior:** per-world, drawn from Dirichlet, **written into the state**; fixed-prior corpus
  kept as a debugging baseline.
- **Likelihoods:** **fixed across the corpus** in v0 (hidden but constant ⇒ recoverable);
  varying-likelihood worlds require exposing/inferring the tables → v1.
- **Score head:** ordinary softmax over levels; class→level map stated in the question.
- **Choice semantics:** include an explicit `OUT` ("outside `S`") option so `y_true` always
  maps to an option and abstention is testable.
- **Training arms:** three — soft `r`, sampled `y~r`, argmax — with the **sampled arm as the
  load-bearing control**. Headline narrowed to "hardening targets harms posterior fidelity."
- **Option-invariance:** guaranteed by an **option-independent read**; verified numerically
  within tolerance, not bit-identical; tested on named classes only.
- **Latent `K` fixed in v0;** vary the *number of presented options* instead (already exercises
  runtime option handling). New class counts are a later test — needs a rule for how the model
  gets evidence meanings for unseen classes.
- **`OUT` = complement mass** `Σ_{k∉S} r_k`; tested separately from named-class invariance;
  means "outside the options," not abstention.

**Still open:**
- How thin v1 text can stay while adding only language difficulty (equivalent templates,
  shuffled evidence order, held-out paraphrases; identical info & targets).

**Resolved in v0 build:**
- **`OUT` mechanism:** *internal full-class marginalization*. The model scores all `K` latent
  classes (`z ∈ ℝ^K`); every question is a **partition** of those classes into answer groups
  (choice: named singletons + an `OUT` group of the omitted; binary: `{B, ¬B}`; score: level
  groups), and the answer logit per group is `logsumexp_{k∈group} z_k` (so `OUT` is the
  logsumexp over omitted classes). Because groups partition all `K` classes, softmax over the
  grouped logits equals the marginalized full-class softmax — named-class odds `= exp(z_a−z_b)`
  independent of the option set. The earlier "absolute-anchor" idea `log(1−Σsoftmax(named))` was
  wrong (a named-only softmax sums to 1 ⇒ `log 0`).

---

## 9. v0 results

Symbolic world `K=8, F=12, V=6`; `Ntr=20000`, `Nte=5000`; targets/labels formed **after**
grouping. `additive` trained to convergence (it is exact-capable); `encoder-qry` at 8 epochs.
Posterior fidelity `KL(r‖p)`/`TV` measured against the **analytic** oracle.

| model | arm | KL(r‖p) | TV | acc | ECE |
|---|---|---:|---:|---:|---:|
| additive | soft | **0.0002** | 0.001 | 0.623 | 0.003 |
| additive | sampled | **0.0041** | 0.028 | 0.619 | 0.002 |
| additive | argmax | **0.6721** | 0.251 | 0.620 | 0.061 |
| encoder-qry | soft | 0.0120 | 0.043 | 0.621 | 0.003 |
| encoder-qry | sampled | 0.1653 | 0.182 | 0.563 | 0.008 |
| encoder-qry | argmax | 0.3844 | 0.229 | 0.598 | 0.052 |

Findings (scoped to this synthetic world):
- **Hardening targets destroys posterior fidelity, and — for the exact-capable additive model —
  does so at matched accuracy.** Additive argmax vs soft: `KL 0.67 vs 0.0002` (~3000×), `ECE
  0.061 vs 0.003`, but **accuracy 0.620 vs 0.623**. The decision is preserved; the probabilities
  are not. This is the narrowed headline, made quantitative.
- **The sampled arm earns the claim.** `sampled` also trains on one hard label per example, yet
  recovers near-ceiling fidelity (additive `KL 0.0041`), while `argmax` does not. So the damage
  is specifically from *choosing the winning label as the target* (erasing the distribution),
  not from single-label supervision per se — the control that makes this more than "erased info
  can't be recovered."
- **Matched accuracy is an additive-model result, NOT universal.** For `encoder-qry`, accuracy
  *does* separate the arms: `sampled 0.563` (−5.8 pts vs soft `0.621`) and `argmax 0.598`
  (−2.3 pts). So accuracy cannot cleanly distinguish models only where they are near the Bayes
  ceiling (additive, all ~0.62); elsewhere it can. Posterior fidelity separates them in *both*
  cases — that is its value, but the earlier "every arm reaches Bayes accuracy" claim was wrong.
- **The encoder's `sampled` gap is a training-efficiency signal, not a ceiling.** At this budget
  (8 epochs) soft targets let the encoder learn faster than the higher-variance sampled signal;
  this does **not** establish that sampled-label training cannot catch up with more data/epochs.
- **Correctness:** `additive/soft` reaching `KL≈0.0002` is *supporting* evidence that the
  eval/marginalization/`OUT` machinery is right — not proof. The independent generator checks
  (§5: tiny-world exact enumeration, oracle reliability) remain the primary correctness gates.
- **Aggregate calibration ≠ posterior fidelity.** `additive/sampled` has slightly *better* ECE
  (0.002) than `additive/soft` (0.003) despite ~20× *worse* KL — ECE can look fine while the
  full distribution is off, which is exactly why we score `KL/TV` to the oracle.
- **Temperature control — partly, not fully.** Best-case `T` (fit to minimize
  `KL(r‖softmax(z/T))` on a held-out split) softens argmax's overconfidence (`T>1`) and removes
  most of the error, but a global rescale cannot close the gap to `soft`:

  | model | argmax KL | argmax+temp KL (T) | soft KL |
  |---|---:|---:|---:|
  | additive | 0.6721 | 0.0233 (T=3.10) | 0.0002 |
  | encoder-qry | 0.3844 | 0.0920 (T=2.08) | 0.0120 |

  So argmax's damage is *mostly* over-sharpness (repairable) **plus** a residual (~100× additive /
  ~8× encoder worse than soft) that rescaling cannot fix — per-instance distortion, not just a
  global scale. Both halves are the useful conclusion.

Reproduce: `python datagen.py selftest` · `python train.py invariance` · `python train.py compare`
· `python train.py tempctl`.

### 9a. Interaction world (`ix`) — does the encoder learn non-additive evidence?

World `K=8, F=15, V=6`: **5 interaction pairs** (joint informative, single-feature marginals
uniform across classes), 4 single slots, 1 noise. `Ntr=40000`, encoder 25 epochs.
Reproduce: `python train.py compare --world ix --ntr 40000 --epochs 25`.

| model | arm | KL(r‖p) | acc |
|---|---|---:|---:|
| additive | soft | 0.6195 | 0.539 |
| additive | argmax | 0.6458 | 0.539 |
| encoder-qry | soft | **0.0377** | **0.776** |
| encoder-qry | argmax | 0.2568 | 0.768 |
| encoder-qry | sampled | 1.5079 | 0.144 |

- **The encoder-query architecture earns its keep — the first place it beats the reference.** It
  recovers the non-factorizable posterior (KL 0.038, near-ceiling; acc 0.776), while the additive
  model — which sums per-feature contributions and *provably cannot represent the joint* — is
  pinned at KL≈0.62 / acc 0.539 **regardless of arm**. No training choice fixes a model that
  can't express the interaction.
- **Under our tested configuration and budget, soft targets enabled learning where `sampled` did
  not.** The encoder `sampled` arm collapsed (acc 0.144 ≈ random, KL 1.51) while soft learned the
  interactions; argmax (a consistent if biased target) learned decently (KL 0.257). *Caveat:*
  `sampled` and soft cross-entropy share the same expected objective, so this is real evidence of
  a **training difficulty**, but **high gradient variance is only a proposed explanation** — not
  established. Controls to run before drawing conclusions: a **label-mapping sanity check** (is the
  sampled label wired to the right answer group?) and **repeats across seeds** (rule out a single
  unstable run / bug). Until then, claim only the conditional statement above.

---

## 10. Roadmap to a real model (B → C → D)

The synthetic stages prove the **method**; a real Jev-like model must **re-demonstrate**
trustworthy confidence — the training method transfers, calibration does not come for free.

**Stage B — same known-posterior world, rendered as text.**
- Render priors + evidence as language (several equivalent templates, shuffled order, held-out
  paraphrases). **No leakage:** the surface must add *no* clue about the hidden answer beyond what
  the symbolic evidence carried. Test on **unseen wording/templates**. Success = recovering the
  *known* probabilities through language.
- Keep the class universe **fixed**. `OUT` marginalization assumes a known, exhaustive universe;
  swapping class IDs for text labels does **not** by itself enable arbitrary new classes —
  generalization to new options is a **separate capability**, out of scope for B.

**Stage C — a real decision task.**
- **Teacher probabilities are estimates, not ground truth.** Soft-target training faithfully
  reproduces a teacher's *miscalibration*; an ensemble may improve targets but does not guarantee
  calibration.
- **Evaluate on independently-verified outcomes** — held-out real cases with observed outcomes or
  independently-checked answers, **never the teacher's own probabilities**. Report **log loss /
  Brier alongside reliability plots**; low ECE alone can hide bad probabilities (v0 §9 already
  showed ECE ≠ posterior fidelity).
- **Teacher distillation is *our* proposed approach, not a claim about Jev.** The public launch
  describes large-model reference probabilities for *evaluation*; it does not disclose how Jev's
  training targets were made.
  [launch](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- **Task: expert routing** (dogfoods the routed-experts program). Predict **each expert's
  probability of meeting the acceptance criterion**, then select by quality *and* cost. These are
  **independent per-expert success probabilities (multi-label Bernoulli), NOT a categorical
  "one correct expert" distribution** — several experts may succeed.

**Stage D — package** as a decision API: state + questions (choice / multi-label / score) →
typed, calibrated answers, with an abstain/escalate path.

**Architectural note (the main model change C introduces):** v0 collapsed every question into a
marginalization of one latent categorical because the synthetic world had a single true class.
Real tasks include **independent** outcomes, so the model needs a second head family —
**independent per-option sigmoids (multi-label)** alongside the softmax-over-partition
(categorical, with `OUT`). Both sit naturally on the shared encoder + option-independent read;
only the output head and target semantics change.

---

## 11. Stage B — execution plan (built, ready to run)

Decision: **ModernBERT-base on the Blackwell box, single GPU** (149M params — no distributed
training). The Mac does data-gen + pipeline checks; the box fine-tunes. Rationale: the symbolic
encoder already showed interaction learning; the open question is whether a *pretrained language
encoder* supports the same probability recovery — a scratch encoder wouldn't answer that.

Files (all built; realizer + pipeline verified on the Mac):
- `realize.py` — text realizer, 3 train templates + 1 held-out; no-leakage/determinism verified.
- `model_text.py` — ModernBERT-base encoder + K class-query reader + typed heads + fixed-class
  `OUT`; encoder fine-tuned alongside the new components (slow LR on encoder, faster on heads).
- `train_text.py` — tokenize → fine-tune → eval; `smoke` (tiny wiring check) and `run`.

Run on the box (identical data across arms; soft first):
```
python train_text.py run --arm soft    --device cuda --ntr 20000 --nte 5000 --epochs 4 --bs 32
python train_text.py run --arm sampled  --device cuda --ntr 20000 --nte 5000 --epochs 4 --bs 32
python train_text.py run --arm argmax   --device cuda --ntr 20000 --nte 5000 --epochs 4 --bs 32
```

Evaluation reports two rows — **unseen states / seen wording** and **unseen states / held-out
templates** — so wording generalization is explicit. Every rendering of a given state stays within
one split (states never cross train/test).

**Success criterion:** low `KL`/`TV` to the oracle on *both* rows (probability recovery survives
language and wording change), with question-isolation and option-invariance still passing. Report
log-loss/Brier + reliability, not ECE alone.

**Boundary held for B:** fixed, exhaustive class universe (required by `OUT` marginalization);
runtime *new* options are a separate capability (Stage C+), not tested here.

**Sampled-collapse follow-up** (before making any claim beyond §9a's conditional): (i) label-
mapping sanity check — confirm the sampled label is wired to the correct answer group; (ii) repeat
the `sampled` arm across ≥3 seeds. If it persists it's a real training difficulty; the
gradient-variance mechanism remains a hypothesis.

### 11a. Stage B result — soft arm (ModernBERT-base, v0 world) ✅

`Ntr=20000`, 20 epochs, warmup+decay, `bert_lr=5e-5`, grad-clip; 1× RTX PRO 4500 Blackwell.

| eval | KL(r‖p) | TV | acc | ECE |
|---|---:|---:|---:|---:|
| unseen states / seen wording | **0.0025** | 0.020 | 0.625 | 0.004 |
| unseen states / held-out templates | **0.0037** | 0.026 | 0.636 | 0.004 |

odds-invariance: **PASS** (1.9e-6).

- **Calibration survives the move to language.** A fine-tuned ModernBERT recovers the exact
  posterior from English prose to `KL≈0.003` — below the symbolic encoder (0.012) and far under
  the ignore-evidence prior baseline (0.605). Stage B's success criterion is met.
- **It reasons, not memorizes.** Held-out-template KL (0.0037) ≈ seen-wording KL (0.0025): wording
  it never trained on barely degrades it.
- **First attempt failed at KL 0.82** (flat LR, 4 epochs — worse than the prior baseline). Pure
  optimization failure; **warmup + real LR/budget fixed it** (loss floor 0.98 → 0.58). A cautionary
  data point: at a bad training config a capable encoder looks like it "can't", when it can.
### 11b. Stage B three-arm comparison (ModernBERT, v0 world, held-out templates)

| arm | KL(r‖p) | acc | ECE |
|---|---:|---:|---:|
| soft | **0.0037** | **0.636** | 0.004 |
| argmax | 0.613 | 0.401 | 0.028 |
| sampled | 0.723 | 0.323 | 0.005 |

- **soft ≫ both single-label arms, robustly (3 seeds).** KL on held-out templates:

  | arm | seed 0 | seed 1 | seed 2 |
  |---|---:|---:|---:|
  | soft | 0.0037 | 0.0051 | 0.0055 |
  | sampled | 0.723 | 0.743 | 1.087 |
  | argmax | 0.613 | 1.106 | 0.860 |

  soft is tiny and stable every time; both single-label arms fail every time (0.6–1.1). The
  firm claim is **soft ≫ {argmax, sampled}**.
- **The sampled-vs-argmax order is NOT robust — it flips across seeds** (seed0 argmax<sampled,
  seed1 sampled<argmax, seed2 argmax<sampled). So the earlier "sampled is worst" was a one-seed
  artifact; both single-label targets are just bad and high-variance. The directional intuition
  still holds — for an expressive model, a single-label target (biased *or* high-variance) trains
  far worse than dense soft targets — but do **not** rank sampled below argmax.
- **ECE ≠ posterior fidelity, vividly:** `sampled` ECE (0.005) beats `argmax` (0.028) yet its KL
  (0.72) is the worst of the three. Aggregate calibration looked fine while the distribution was
  far off — the reason we grade on KL/TV to the exact answer.
- **Firm takeaway:** soft/distributional targets recover calibrated confidence from language
  (KL 0.004); both single-label alternatives fail. For Stage C this argues strongly for teacher
  *soft* targets over training on single observed outcomes (which is the `sampled` regime).
- *Caveats:* the soft≫single-label result is confirmed over 3 seeds; the sampled/argmax *ordering*
  is not robust (see above). "single-label fails" is at this budget (same expected objective ⇒ more
  epochs/data could narrow it). Label-mapping check passed (the sampled weakness is a real training
  difficulty, not a wiring bug).
  *(Label-mapping check since ran and PASSED — every true class lands in exactly one answer group,
  soft target normalized, sampled label wired to the correct group — so the sampled weakness is a
  real training difficulty, not a wiring bug. Seed repeats in progress.)*

---

## 12. Stage C (step 1) — controlled multi-label expert routing ✅

First real *decision* task, still in the known-answer world so we can grade exactly before
touching real teacher data. `N=6` experts, each with a fixed competence `C[e,k]`; given a state
the exact per-expert success prob is `s_e = Σ_k r_k·C[e,k]` — **independent Bernoullis, not a
simplex**. Same ModernBERT encoder + query reader, but a **per-expert sigmoid head** (the
multi-label capability §10 flagged), trained with BCE. Metrics vs exact `s`: MAE, Brier; **routing
regret** in true-success units (0 = picks the oracle's expert), plain and cost-aware (`λ=0.3`);
ECE vs sampled outcomes. `train_route.py`, 15 epochs, held-out templates.

| arm | MAE | Brier | regret | regret+cost | ECE |
|---|---:|---:|---:|---:|---:|
| soft | **0.017** | **0.0007** | **0.0014** | **0.0012** | 0.007 |
| sampled | 0.137 | 0.032 | 0.083 | 0.070 | 0.008 |
| argmax | 0.290 | 0.102 | 0.054 | 0.059 | 0.291 |

- **The multi-label head works.** soft predicts per-expert success to ~1.7% MAE from English and
  routes near-optimally (regret 0.0014).
- **Calibration ≠ decision quality — cleanly separated here.** argmax has the *worst* probabilities
  (ECE 0.29, from 0/1 targets) yet *better* routing than sampled (regret 0.054 < 0.083): thresholding
  preserves the expert *ranking*, while sampled's high-variance targets corrupt even the ranking.
- **Cost-aware routing requires calibration.** On regret+cost, soft (0.0012) dominates argmax
  (0.059) and sampled (0.070). Bare argmax routing tolerates junk probabilities; trading
  success-vs-cost, or abstaining on low confidence, does not — **this is the justification for a
  calibrated decision model.**
- Consistent with §11b: soft ≫ the single-label arms on fidelity; the `sampled` high-variance
  target underfits the expressive model.
- *Caveats:* controlled synthetic (known answer), n=1 seed, **fixed expert universe**. The real
  Stage C still needs LLM-generated states + **runtime experts with text descriptions** + teacher
  *soft* targets + eval on **independent outcomes** (§10) — this step validates the head, the
  objective, and the routing metric, not the real-data pipeline.

---

## 13. Generic runtime-option model — does it generalize to unseen options? ✅

The routing model (§12) still used a fixed option set. The **generic** model drops that: each option
is defined by a runtime **description** — a *prototype* ("what evidence this option typically
produces") — encoded and scored **independently** against the state (`genworld.py`, `model_generic.py`,
`train_generic.py`). An unseen option is just a new description. Soft targets, ModernBERT, 30 epochs.

| eval (held-out states) | KL(r‖p) | TV | acc | ECE |
|---|---:|---:|---:|---:|
| unseen options — new prototypes + names + wording | **0.128** | 0.089 | 0.818 | 0.012 |
| + more options — 8–10 presented, trained on 3–6 | 0.217 | 0.137 | 0.725 | 0.010 |

option-order invariance: **0.00** (exact, structural).

- **Runtime options generalize.** The model handles options/prototypes/names/wording it never trained
  on, with good calibration (KL 0.13, ECE 0.012) and 0.82 accuracy — it learned the *rule* (match
  evidence to each option's stated profile), not fixed class identities. This is the capability a
  *generic* decision primitive needs.
- **Extrapolating to more options** than ever seen (8–10 vs 3–6) degrades but holds (KL 0.22, acc
  0.73) — an honest limit.
- **Genericity costs some fidelity** vs the fixed-universe setting (Stage B KL 0.004): reading
  arbitrary descriptions + matching is a harder computation. Reasonable trade for runtime options.
- **Undertraining, a third time:** 15 epochs gave KL 0.42; 30 epochs → 0.13 (loss 0.48→0.33). With
  this method, apparent calibration ceilings are usually optimization budget, not capability.
- **Order-invariance is exactly 0** — options are scored independently (the property GLiClass's
  uni-encoder trades away for efficiency). Related work: GLiClass/GLiNER/SemIf nail the runtime-label
  *mechanism*; certo adds *calibration by construction* on top.
