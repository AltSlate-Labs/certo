# Data preparation for a general-purpose decision model

**Reviewed plan · 20 September 2026**

**Goal:** train a reusable model that reads a state, a question, and supplied criteria, then returns typed decision probabilities. The first milestone is transfer across tasks and unfamiliar criteria, followed by confidence that is useful on independently evaluated outcomes.

This is a preparation specification for our model. It is not a reconstruction of Jev's undisclosed training pipeline. Existing experimental results are taken from the project updates in our conversation; the implementation and datasets have not been independently audited here.

## 1. Review of the earlier recommendation

The basic direction remains sound: combine real language data, tasks with explicit rules, and controlled probability problems. Several details need correction before building the corpus.

| Earlier suggestion | Reviewed decision | Why it matters |
|---|---|---|
| Start with a 100,000-example mixture | Treat 100,000 as a pilot budget; first validate a small preparation sample | Neither the size nor the percentages are an established optimum |
| Include MultiNLI and ANLI by default | Use MultiNLI as the initial NLI source; exclude ANLI from the default product-training build | ANLI's repository states a noncommercial license; the earlier recommendation omitted this restriction |
| Add Tasksource as another large source | Use Tasksource as a task catalog and optional adapter reference | It already includes MultiNLI, ANLI, and many other datasets; indiscriminate mixing can duplicate data and obscure original splits |
| Convert everything into one-choice classification | Preserve categorical, independent binary, and ordinal semantics | Several labels can be valid simultaneously; forcing one winner changes the task |
| Use soft targets to obtain trustworthy probabilities | Preserve the distinction between exact posteriors, observed labels, and teacher estimates | Only the controlled world supplies the exact posterior; soft teacher targets can transfer teacher errors |
| Test unfamiliar names and templates | Also reserve entire criteria, rule compositions, and task families | Surface variation alone does not establish general decision ability |
| Treat missing information as uncertainty | Define whether the task asks for truth, evidential support, or the next action | “Ask for information” is a decision; it is not automatically a 50/50 truth probability |
| Train directly from dataset records | Build explicit input-field allowlists | Some source records contain explanations or answer-derived annotations that would leak the target |

The ANLI restriction is documented in its [official repository](https://github.com/facebookresearch/anli#license). The [Tasksource viewer](https://huggingface.co/datasets/tasksource/tasksource-instruct-v0) visibly includes both MultiNLI and ANLI tasks. Those are concrete reasons to revise the source list.

**First implementation:** direct source adapters, an executable-policy generator, the existing posterior generator, and a shared data validator. Teacher-generated probability targets are an optional later addition, not a prerequisite.

## 2. Recommended initial corpus

### 2.1 A concrete 100,000-record starting allocation

Counts below mean canonical **state–question records after splitting and filtering, before paraphrase expansion**. They are engineering budgets. Validation, calibration, and test records are additional and never count toward these training totals.

| Family | Training records | Initial composition | Main purpose |
|---|---:|---|---|
| Supplied rules and rubrics | 30,000 | Own generated policies, workflow decisions, comparisons, and scoring rubrics | Follow criteria that change between examples |
| Evidence-based inference | 25,000 | MultiNLI training data | Distinguish support, contradiction, and insufficient evidence |
| Language classification | 25,000 | 10,000 CLINC150; 5,000 BANKING77; 10,000 GoEmotions | Understand category descriptions and multiple valid labels |
| Multiple-choice reasoning | 10,000 | 8,000 PIQA; up to 2,000 ARC across available training subsets | Compare plausible candidate answers |
| Known-posterior decisions | 10,000 | Existing generator, including interaction worlds and text rendering | Retain an exact probability reference |
| **Total** | **100,000** | | |

If filtering or held-out categories leave too few records, reduce the total or replace records from another eligible source in the same family. Do not duplicate examples to reach a round number or borrow validation/test rows. Record actual counts.

The classification slice is initially concentrated in intents and expressed emotion. It does not establish broad domain coverage by itself. The generated slice should cover several settings: support, document workflows, education exercises, inventory, incident triage, and expert routing simulations. Later add owned or otherwise eligible real examples in those settings.

Use the quotas as initial sampling proportions, with a cap on contributions from any one root state. Record both question counts and independent root-state counts. Ten questions about one record are ten supervised questions, not ten independent situations.

### 2.2 Source decisions and preparation notes

| Source | Initial role | Preparation decision |
|---|---|---|
| [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/) | Core training | Keep premise and hypothesis; preserve its three-way meaning. Record genre and source identifiers outside model input. Its release contains multiple source licenses, so retain the release notices and provenance rather than assigning one invented blanket license. [Release card](https://huggingface.co/datasets/nyu-mll/multi_nli) |
| [CLINC150](https://github.com/clinc/oos-eval) | Core intent training | Single-intent examples; explicit out-of-scope data. Its repository provides separate train, validation, and test partitions. Preserve them. The [published license](https://github.com/clinc/oos-eval/blob/master/LICENSE) is CC BY 3.0. |
| [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) | Core fine-grained intent training | Useful for distinguishing nearby categories. Use the supplied training/test distinction; create development partitions from training groups. The repository publishes CC BY 4.0 terms for its datasets. |
| [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) | Core multi-label training | Use the simplified release and its supplied splits initially. Preserve multiple annotations as multiple labels; do not softmax them into one winner. The release card identifies Apache 2.0 repository terms. |
| [PIQA](https://yonatanbisk.com/piqa/) | Core candidate comparison | Preserve the goal and both candidate solutions. The project publishes AFL 3.0 terms. |
| [ARC](https://huggingface.co/datasets/allenai/ai2_arc) | Small reasoning supplement | Preserve the original choice texts and answer-key mapping. The release card identifies CC BY-SA 4.0. |
| [ShARC](https://sharc-data.github.io/data.html) | First real-policy extension | Add after its conversion is validated; replace part of the 30,000-record policy budget rather than silently enlarging it. The data page specifies CC BY-SA 3.0 and the permitted input fields. |
| [HellaSwag](https://rowanzellers.com/hellaswag/) | Initially reserve from decision-model training | Useful as a different candidate-comparison task. If already used for training, move it to the seen-task evaluation track and reserve another source. |
| [ContractNLI](https://stanfordnlp.github.io/contract-nli/) | Later long-document evaluation | A useful evidence test, but its hypotheses are fixed across documents. It is not by itself a test of entirely new question semantics. The project provides CC BY 4.0 terms. |
| [ANLI](https://github.com/facebookresearch/anli) | Excluded from default product-training build | The published noncommercial restriction makes it unsuitable for unconditional inclusion. Any separate use requires a basis consistent with those terms. Do not infer permission from a copy inside an aggregate. |
| [Tasksource](https://huggingface.co/datasets/tasksource/tasksource-instruct-v0) | Discovery and adapter reference | Select tasks explicitly, resolve their upstream source and split, and deduplicate against direct imports. Do not import the whole mixture by default. |

The license entries report published source information, not a blanket clearance of every underlying text or intended use. Save the exact release notices with each snapshot. A wrapper's software license does not establish the rights to every included dataset. In particular, the [AG News card](https://huggingface.co/datasets/fancyzhx/ag_news) lists an unknown license and describes noncommercial source use; leave it outside this default build while that remains unresolved.

## 3. Specify what each probability means

Each task needs an **event definition** before it needs a label.

| Output type | Meaning | Example | Target representation |
|---|---|---|---|
| Choice | Exactly one answer under the stated task convention | Which supplied policy action applies? | One class or a distribution over mutually exclusive options |
| Binary | Whether one defined event holds | Does the passage support this claim? | A yes/no label or probability |
| Independent binary set | Several events may hold together | Which emotions are expressed? Which experts meet the quality bar? | One Bernoulli target per event, with an annotation mask |
| Score | A level under an explicit ordered rubric | Priority level 0–3 under the provided rules | One level or a distribution over levels |

The model's probability of a category being the annotated answer is different from the probability that an underlying real-world event will occur. Preserve that distinction in instructions and evaluation.

For example:

- “Is this claim supported by this document?” concerns evidence.
- “Is this claim true in the world?” can require knowledge absent from the document.
- “Should the system ask for more information?” concerns an action policy.

Never silently substitute one for another. If an ontology permits overlapping categories, use independent binary outputs or explicitly define a tie-breaking rule for a choice task.

### OUT, out-of-scope, and missing information

- **OUT relative to a supplied subset:** the correct class is one of the known classes omitted from that subset.
- **Out-of-scope:** the example falls outside the supported ontology.
- **Insufficient evidence:** the input does not establish the answer under the question's evidence standard.
- **Escalate:** an action selected under a quality/cost/risk policy.

These are different targets. The existing full-class OUT marginalization handles the first case when a complete internal class universe exists. It does not automatically implement the other three or cover arbitrary unseen classes.

## 4. Canonical record format

Store one question per canonical JSONL record, linking repeated questions to the same state through `state_id`. The loader may group these for shared state encoding.

Only `model_input` is visible to the model. Everything else is used for targets, filtering, grouping, or audit. Construct model inputs through an allowlist; never serialize the entire record into a prompt.

### 4.1 Complete illustrative record

The identifiers below are illustrative, not records from an existing corpus.

```json
{
  "schema_version": "decision-data-v1",
  "record_id": "policy-000017-q1-v0",
  "state_id": "policy-000017",
  "split_group_id": "policy-case-000017",
  "task_family": "supplied_policy",
  "model_input": {
    "state": "The item is unused. The return was requested 10 days after purchase.",
    "question": {
      "type": "choice",
      "instructions": "Select the action required by the supplied return policy.",
      "rubric": "Accept an unused item if the return was requested within 14 days, inclusive. Decline when a stated condition fails. Request information only when eligibility cannot be determined from the stated facts.",
      "options": [
        {"id": "a", "description": "Accept the return under the policy."},
        {"id": "b", "description": "Decline the return under the policy."},
        {"id": "c", "description": "Request information needed to determine eligibility."}
      ]
    }
  },
  "target": {
    "kind": "categorical_label",
    "label_id": "a",
    "source_type": "program_verified"
  },
  "metadata": {
    "split": "train",
    "source_id": "own_policy_generator",
    "source_revision": "policy-generator-v1",
    "source_row_id": "policy-000017-q1",
    "original_split": "generated_train",
    "domain": "returns",
    "rule_family": "conjunction_and_threshold",
    "rule_signature": "unused_AND_days_LE_limit",
    "template_id": "return-template-02",
    "label_schema_id": "return-action-v1",
    "annotation_status": "verified",
    "oracle_audit_ref": "policy-000017-oracle",
    "render_seed": 1701
  }
}
```

Keep the rule program, hidden variables, source annotation explanations, and solver trace in a separate audit store. If a rule or prior is required to solve the task, expose its intended textual representation in `model_input`; its existence in metadata alone is insufficient.

### 4.2 Other target forms

Exact categorical posterior:

```json
{
  "kind": "categorical_distribution",
  "probabilities": {"a": 0.7, "b": 0.2, "c": 0.1},
  "source_type": "exact_posterior",
  "oracle_version": "interaction-world-v1"
}
```

Independent binary labels with one unannotated event:

```json
{
  "kind": "bernoulli_labels",
  "values": {"gratitude": 1, "anger": 0, "confusion": null},
  "observed_mask": {"gratitude": true, "anger": true, "confusion": false},
  "source_type": "human_annotation"
}
```

Here, zero means an annotated negative under the source convention. Null means unobserved. Missing annotations are not automatically negative examples.

Teacher distributions use the same shape as the corresponding categorical or Bernoulli distribution, but set `source_type` to `teacher_estimate`. Record teacher identity/version, prompt hash, inference settings, raw response location, and any aggregation/calibration method in metadata. An invalid teacher response must not become a uniform target.

Scores use categorical targets over ordered level IDs, with each level's meaning in the rubric. Do not normalize independent Bernoulli probabilities to sum to one.

## 5. Public-dataset adapters

### 5.1 MultiNLI: evidence-based inference

Use the premise as state and place the hypothesis in the question. Define the options as **supported**, **contradicted**, and **not determined by the premise**. Resolve source label IDs from release metadata rather than assuming numeric order. Missing source labels are excluded, not guessed.

Keep all pairs sharing a premise in one project split when carving development partitions from training. Preserve the original evaluation partitions separately. Include an input-only diagnostic with the premise removed to detect hypothesis shortcuts. [Source](https://cims.nyu.edu/~sbowman/multinli/)

Binary conversions must name the event precisely: “supported?” can map both contradiction and neutral to no; “false?” cannot use that same mapping.

### 5.2 Intent datasets: descriptions and candidate construction

Build a versioned label dictionary from source definitions and training examples only. Each entry needs a description, boundaries against nearby categories, and optional training-derived examples. Human-review the dictionary once; freeze it before evaluation.

For candidate-subset training, include the gold category plus a mix of nearby and unrelated alternatives. Keep the gold present unless an explicit valid OUT mapping exists. Changing the candidate set changes the question, so log the candidate policy.

CLINC's out-of-scope examples mean outside its complete supported intent ontology. They are not interchangeable with an in-scope intent omitted from a sampled subset. [CLINC source](https://github.com/clinc/oos-eval)

BANKING77 provides fine distinctions within one domain; it is useful for semantic boundaries, but does not by itself establish domain transfer. [BANKING77 source](https://github.com/PolyAI-LDN/task-specific-datasets)

### 5.3 GoEmotions: preserve overlapping labels

Use comment text as state and independent binary questions about expressed emotion. The simplified release supports multiple labels and has predefined splits. Preserve those labels; do not reinterpret a multi-label row as a categorical distribution. The question concerns annotation of the text, not a diagnosis of the author's internal state. [Source](https://huggingface.co/datasets/google-research-datasets/go_emotions)

Use the source's annotation convention to determine negatives. If later using raw rater data, distinguish non-selection from “too unclear to label.” Rater frequencies estimate an annotation process and have finite-sample uncertainty; they are not exact world posteriors. Group all annotations by comment before splitting.

### 5.4 PIQA and ARC: retain the original decision

Preserve goal/question text, all original candidates, and the answer mapping. Any option permutation must permute the label mapping identically. Keep source IDs and original splits. Generated explanations, answer keys, and teacher solutions are never input evidence unless a separately named task deliberately supplies them to every model.

Use original candidates for the first build. Adding plausible distractors can create another correct answer; it is a new labeling problem, not harmless augmentation. [PIQA](https://yonatanbisk.com/piqa/), [ARC](https://huggingface.co/datasets/allenai/ai2_arc)

HotpotQA should have a separate future adapter for evidence-grounded answering or answer verification. It is not natively the same multiple-choice task; automatic distractor construction would change its benchmark meaning.

### 5.5 ShARC: a decision adapter, not full question generation

Allowed source input fields are `snippet`, `question`, `scenario`, and `history`. The data documentation explicitly excludes `evidence` and `answer` from input. Map terminal yes/no/irrelevant outcomes to corresponding actions; map a follow-up-question answer to `request_information`. This tests the next decision, not the quality of a generated follow-up question. Keep the original follow-up text in audit metadata only. [Data specification](https://sharc-data.github.io/data.html)

Group related examples by `tree_id`, and audit shared snippets/source documents across evaluation groups. Report this conversion as a custom decision task; do not label its four-action accuracy as the complete original benchmark score.

## 6. Generate policies and rubrics from executable specifications

This generator should supply varied decision boundaries, not merely varied nouns.

### 6.1 Initial rule families

| Family | Example | Necessary variations |
|---|---|---|
| Numeric and date thresholds | Request within a stated deadline | Just below, at, and above the threshold; explicit units and inclusive boundaries |
| Conjunction/disjunction | Eligible if A and B; alternate route if C | Cases isolating each condition and both together |
| Exceptions and precedence | Standard rule with a named override | Override active/inactive; explicit priority when rules conflict |
| Temporal state | Latest valid status determines action | Old versus current evidence; explicit timestamps and tie rules |
| Comparison and selection | Choose an eligible item minimizing stated cost | Close alternatives, ties, and explicit tie-breaking |
| Evidence sufficiency | Ask only for facts needed to decide | Irrelevant missing facts versus decisive missing facts |
| Rubric scoring | Score 0–3 according to supplied criteria | All levels, boundary cases, changed level definitions |
| Multiple independent conditions | Several experts meet an acceptance bar | Zero, one, or several successful candidates |

### 6.2 Generation procedure

1. **Sample a task specification.** Choose domain, rule family, thresholds, exceptions, output type, and event definition.
2. **Sample structured facts.** Include difficult boundaries and counterfactual partners. Use explicit distributions if probabilistic labels will be needed.
3. **Compute targets with an independent evaluator.** For discrete rules, enumerate small cases or execute the rule AST. Validate the evaluator against hand-calculated examples.
4. **Assign the whole case group to a split.** Related states, counterfactuals, and later renderings inherit this assignment. Reserve rule families/compositions separately where required.
5. **Render the observable facts and rules.** Start with deterministic renderers, then introduce LLM paraphrases.
6. **Verify semantic preservation.** Check entities, negation, numbers, units, dates, logical connectors, exceptions, and information presence. Quarantine ambiguous rewrites.
7. **Generate several questions where useful.** Choice, binary, and score views may share a state, but their event definitions remain explicit.
8. **Write model records and audit records separately.** Keep hashes linking them.

For fully specified deterministic rules, a one-hot target is appropriate: the correct rule outcome is deterministic. This is different from replacing a genuinely uncertain posterior by its argmax.

### 6.3 Missing facts require a defined target

For a logical task, enumerate completions compatible with the input:

- All completions satisfy the criterion: established yes.
- All violate it: established no.
- Both are possible: not established; ask for information if the action policy says to do so.

This logical test does not assign probabilities to completions. To produce a probability, define a distribution over missing facts and marginalize it. Uniformly weighting completions is an additional modeling assumption that must be stated or learnable from the training world.

### 6.4 Keep the LLM's role bounded

Use a teacher to generate natural phrasing, plausible contexts, and candidate descriptions. The source of correctness for executable tasks remains the evaluator.

An LLM renderer should receive only the information intended to be expressed. Hidden labels or hidden facts can leak through phrasing even if their field names are removed. Separate observable-state rendering from answer computation wherever possible. Use deterministic templates for the initial exact-posterior evaluation.

A semantic-review prompt can help filter prose, but another teacher saying “looks correct” is not a proof of equivalence. Use structured checks where possible and inspect a stratified sample of accepted rewrites.

## 7. Preserve the exact-posterior reference

Let `u` contain the underlying observable facts and `t` their text rendering. Reusing the original oracle requires:

\[
P(Y\mid t,q,O)=P(Y\mid u,q,O).
\]

This holds when the rendering preserves all relevant observable information and contributes no additional information about the hidden answer. If rendering drops evidence, rounds a consequential prior, introduces ambiguity, or hints at the answer, the old posterior may no longer be the correct target for the text.

For each known-posterior record:

- Include the prior and evidence needed by the model, following the established world's identifiability assumptions.
- Record the generator version, parameter regime, and random seed.
- Retain the exact distribution in audit/target fields only.
- Check normalization, nonnegativity, finite values, and tiny-world enumeration.
- Preserve coverage across posterior entropy, evidence strength, interactions, and option counts.
- Recompute the answer distribution when grouping or changing options.

For a fixed exhaustive class universe with posterior `r`, a subset `S` plus OUT has:

\[
r_{\mathrm{OUT}}=\sum_{k\notin S}r_k.
\]

For a task explicitly conditioned on the answer being in `S`:

\[
r'_k=\frac{r_k}{\sum_{j\in S}r_j},\qquad k\in S.
\]

These are different tasks. Reject the conditional construction if its conditioning event has zero probability. For grouped score levels, sum the underlying class probabilities within each level.

Do not claim exact posterior fidelity for real annotated datasets merely because their gold label can be written as a one-hot vector.

Also preserve the sampling assumptions behind the oracle. Selecting cases by observable features or oracle entropy changes coverage without changing the conditional label distribution at a given input. Selecting by the sampled hidden label can change that conditional distribution. For class-dependent acceptance rates `a_k`, the selected population has posterior proportional to `a_k * r_k`. Either recompute that posterior or avoid this form of selection in the exact-target track.

## 8. Label provenance and training objectives

| Target source | What is known | Training use | Evaluation interpretation |
|---|---|---|---|
| Program-verified deterministic label | Rule outcome for the stated task | Ordinary categorical or binary loss | Correct execution of that rule |
| Human/observed label | One annotation or outcome | Ordinary supervised loss | Predictive quality against the chosen annotation/outcome standard |
| Exact posterior | Full distribution under a specified generator | Soft cross-entropy | KL/TV to that oracle, plus decision quality |
| Teacher estimate | A model's estimated probabilities | Optional distillation loss | Teacher agreement; outcome evaluation is still required |
| Rater distribution | Empirical annotation frequencies | Optional distributional supervision | Agreement with an annotation population, with sampling uncertainty |

For categorical predictions:

\[
\mathcal L_{\mathrm{gold}}=-\log p_y,
\qquad
\mathcal L_{\mathrm{soft}}=-\sum_a r_a\log p_a.
\]

When labels are sampled from the actual conditional distribution, the expected gold-label loss equals soft cross-entropy. A collapse in one sampled-label experiment does not make all human-labeled datasets unsuitable.

For independent binary events, use binary cross-entropy per annotated event, averaging over the observed mask. For rubric scores, use cross-entropy over the defined levels initially. A teacher distribution `q` may replace `r` in a separate distillation term, but it must retain its teacher provenance.

Normalize loss per question; for a multi-label question, average across annotated events. Then average questions within a state or cap their contribution so states with many generated questions do not dominate accidentally. Log both the sampling rule and loss weighting.

**Default pilot:** gold/program labels plus exact posteriors, with no teacher probability supervision. Add teacher targets only as a named ablation after measuring this baseline. This makes their incremental value observable.

## 9. Split before augmentation

### 9.1 Maintain distinct evaluation tracks

| Track | What is held out | What it can establish |
|---|---|---|
| Seen-task generalization | New states in trained task families | Generalization within a known task |
| Surface transfer | Names and renderer/template families | Robustness to wording and naming |
| Criteria transfer | Label sets, rule structures, or rule compositions | Application of unfamiliar decision criteria |
| Domain/task transfer | Entire sources or semantic task families | Transfer beyond the training task distribution |
| Cardinality transfer | Larger/smaller candidate sets | Robustness to changes in option count |
| Deployment evaluation | Independent cases from intended use | Quality and confidence where the model will be used |

Dataset holdout is not always task-family holdout: reserving a second NLI corpus still tests a familiar NLI task. Report these separately. Public benchmark holdout from this fine-tuning corpus also does not establish absence from the pretrained encoder's earlier training.

### 9.2 Project split policy

- Preserve source train/dev/test labels. Never move an official test example into project training.
- From eligible source training groups, reserve a reproducible development set for model selection and a separate calibration set for fitting temperature or thresholds. A starting allocation is 90% train, 5% development, 5% calibration; adjust small-source counts explicitly.
- Keep official validation/test partitions for their declared evaluation role. If there is no separate accessible test, reserve a final project test from training before tuning and label it as a project test.
- Synthetic groups can use 80% train, 10% development, 5% calibration, 5% final test, in addition to completely held-out rule and renderer families. These are starting allocations, not statistical guarantees.
- Treat prior repeatedly inspected test sets as development evidence. Generate or reserve a fresh final set for the next frozen model comparison.

Use stable hashes of explicit group IDs. Groups must connect all paraphrases, option permutations, counterfactual partners, and questions derived from one root situation. NLI premises, source documents, dialogue histories, and raw annotation IDs can require larger groups.

A duplicate cluster that crosses an official training/test boundary stays out of training; preserve and report the official test membership. Similarity-based near-duplicate matches require inspection because a changed negation or threshold can be a legitimate counterfactual.

For label-held-out tests, remove those labels' examples from training and their descriptions from training candidate lists. Freeze test descriptions from independently defined ontology text; do not derive them from test examples. If unlabeled text reveals the held-out category, document that limitation.

Keep an explicit record of which task/domain holdouts are used for development and which are final. Choosing mixtures repeatedly on the same held-out family makes that family part of model development, even when its individual examples never receive gradient updates.

## 10. Option construction and semantic augmentation

### 10.1 Transformations that should preserve the answer

- Permute options and update target-ID mapping.
- Rename entities consistently where their identities are irrelevant.
- Paraphrase instructions without changing the evidence standard.
- Reorder facts only when chronology and interpretation are unchanged.
- Add genuinely irrelevant information.

### 10.2 Transformations that require a new target

- Change a policy threshold, exception, priority rule, or score rubric.
- Remove or alter relevant evidence.
- Change candidate meanings or add another plausible answer.
- Change whether the question concerns support, truth, or action.
- Omit the correct candidate and introduce a defined OUT option.

Where a solver exists, recompute. For ordinary real examples, retain only label-preserving transformations whose semantics have been checked, or obtain a new annotation.

An initial candidate-count schedule for tasks that support it is 2–4, 5–8, and 9–16 options, with 17–32 reserved as an extrapolation test. This is a proposed coverage schedule. Do not manufacture extra options for native two-choice tasks merely to fill a bucket.

For a selected intent-label subset, use approximately half plausible neighboring negatives and half broadly sampled negatives as a starting recipe. Balance lexical style and description length across correct and incorrect candidates. Use only training-side evidence to construct these alternatives.

### 10.3 What structural invariance does and does not mean

Permutation invariance means reordering the same candidate meanings preserves their predictions after remapping. Independent option scoring can additionally preserve pairwise odds when other independent options are added.

That stronger property is appropriate only when the event definitions and required model assumptions support it. Overlapping categories, context-dependent comparisons, and “best of this set” questions can make strict option independence too restrictive. Record the intended semantics rather than enforcing one invariance test on every task.

## 11. Validation before training

Run these checks on the prepared corpus. Fix the underlying adapter or generator when a systematic failure appears; do not patch individual labels without recording the correction.

| Check | Required behavior |
|---|---|
| Schema | All records parse; required fields and output types agree |
| Label IDs | Every categorical target maps to a supplied option; IDs are unique |
| Distributions | Nonnegative, finite values; categorical sums within `1e-6` of one; Bernoulli values in `[0,1]` |
| Multi-label masks | Unknown annotations are excluded from the loss; positives are not forced to sum to one |
| Input-field boundary | Targets, source explanations, solver traces, and answer-derived annotations never enter `model_input` |
| Split lineage | No known root group or duplicate cluster appears on both sides of a train/evaluation boundary |
| Source mapping | Original labels and splits survive the adapter; all exclusions have reasons |
| Rendering | Relevant facts, rules, priors, and units remain recoverable; no answer hints are introduced |
| Tokenization | No silent truncation of decisive evidence, rule clauses, or options |
| Oracle | Program labels agree with an independent small-case check; posterior marginalizations agree with enumeration |
| Metamorphic pairs | Paraphrases preserve targets; controlled semantic changes produce the evaluator's new targets |
| Coverage | Report labels, domains, task families, option counts, lengths, missingness, and posterior entropy |

Start with a stratified manual review of roughly 200–300 prepared records, covering every adapter, output type, and difficult generator family. This is a bug-finding sample, not proof of a negligible error rate. For each systematic error found, fix the converter and check fresh examples from the affected category.

Token-length handling must be decided before export: use evidence-preserving windows for a named windowed task, route long records separately, or exclude and report them. A truncated input must not inherit a target that depended on removed evidence.

## 12. Data-quality diagnostics and evaluation

### 12.1 Cheap diagnostics worth keeping

- **Question/options-only baseline:** remove the state. High performance can reveal label style or question shortcuts.
- **Shuffled-state baseline:** preserve questions but swap states within a compatible task family. A small drop suggests weak use of evidence.
- **Counterfactual pairs:** measure whether the answer changes when the decisive rule/fact changes, as well as whether each answer is correct.
- **Description sensitivity:** paraphrases should preserve meaning; changing a description's meaning should change the corresponding decision when appropriate.

Use these diagnostics to investigate concrete failures. Passing them does not certify general reasoning.

### 12.2 Metrics by target type

| Evaluation data | Main measurements |
|---|---|
| Exact posterior | Mean KL in nats, TV, accuracy, oracle accuracy, accuracy regret, and entropy/cardinality slices |
| Real categorical outcomes | Accuracy, NLL, multiclass Brier score, reliability plots, ECE with stated bins |
| Independent binary outcomes | Per-label and macro Brier/log loss; precision/recall at thresholds fixed on development/calibration data |
| Ordinal levels | Level NLL/Brier plus absolute level error under the stated numerical rubric |
| Selective automation | Error rate versus fraction automatically handled; actual escalation cost when available |

For the synthetic choice task, report `mean(max(r))` as oracle accuracy and its gap to model accuracy. Random-guess comparisons alone do not separate model error from intrinsic ambiguity.

Specify metric conventions: use `TV = 0.5 * sum(abs(r-p))`; report multiclass Brier as `sum((p-one_hot(y))^2)` averaged across examples. For independent binary targets, average squared error over observed labels. Avoid directly comparing aggregate scores across very different task mixes or option counts without per-task results.

Fit any temperature scaling on the calibration partition only. Report uncalibrated and calibrated results on the untouched evaluation set. A global temperature can be a baseline; it does not guarantee calibration on new task families.

Balance task exposure for learning, but preserve representative evaluation prevalences. Class balancing and forced candidate selection can change the data distribution and the probability being estimated. Record those choices rather than presenting confidence from a balanced benchmark as deployment confidence.

Use confidence intervals or bootstrap intervals clustered by root state/document, and compare models on the same cases. Aggregate calibration can hide poor performance on rare labels or difficult tasks; retain per-family reports.

## 13. Reproducible preparation pipeline

### 13.1 Files to implement

These are proposed project files, not an assertion that they already exist.

| File/module | Responsibility |
|---|---|
| `data/registry.yaml` | Sources, release revisions, hashes, published terms, eligible configurations, and permitted model fields |
| `data/task_specs.yaml` | Task semantics, label dictionaries, rubrics, candidate policies, and held-out families |
| `prepare/snapshot.py` | Fetch pinned releases and retain original splits and notices |
| `prepare/adapters/` | One explicit adapter per source |
| `prepare/policies.py` | Structured rule/fact generation and independent label evaluation |
| `prepare/render.py` | Deterministic rendering and optional verified teacher paraphrases |
| `prepare/split.py` | Root grouping, duplicate resolution, and frozen split assignment |
| `prepare/validate.py` | Schema, leakage, target, oracle, and token-length checks |
| `prepare/export.py` | JSONL/Parquet exports and model-input-only views |
| `reports/data_card.md` | Counts, exclusions, source lineage, coverage, audit findings, and limits |

### 13.2 Registry fields

For each source, record:

`source_id`, canonical URL, configuration, release/revision, retrieved date, file hashes, original split names, label mapping, published license/terms reference, intended-use status, allowed input fields, prohibited annotation fields, grouping keys, adapter version, and exclusion rules.

For teacher-produced examples, additionally retain model/version, prompt hash, inference parameters, generation timestamp, parent source IDs, validation status, and cost. For generated examples, retain generator/solver versions and seeds.

Build order:

1. Freeze task semantics and the source registry.
2. Snapshot eligible raw sources without modifying their original files.
3. Resolve grouping and duplicate relationships; reserve evaluation/task holdouts.
4. Convert through explicit adapters and generate structured synthetic cases.
5. Assign project splits before paraphrase or option augmentation.
6. Render, recompute targets where needed, validate, and audit.
7. Select canonical training records up to the family budgets.
8. Export immutable versioned partitions plus a build manifest and data card.

A preparation run must log input hashes, configuration hashes, random seeds, code revision, accepted/rejected counts by reason, unique root counts, and output hashes. Repeating the build with the same frozen inputs and deterministic settings should reproduce the same output. Cache stochastic teacher outputs as immutable inputs to later rebuilds.

## 14. Execution milestones

### Milestone 1 — Validate the preparation system

Prepare approximately 2,000–5,000 records spanning all core adapters and output types. Include boundary cases and negative examples deliberately. Run the validation suite and manual review; verify a small model batch can consume every target type. Do not interpret this as a capability benchmark.

### Milestone 2 — Run the first mixed-data pilot

Build the frozen 100,000-record target corpus, or the smaller actual total after filtering. Keep the existing model architecture and record all initialization and training-budget choices. Select checkpoints using development metrics, not the final test.

Begin with gold/program labels and exact posteriors. Judge improvement by per-task predictive quality and criteria transfer. Synthetic KL remains a separate reference, not a replacement for real-data evaluation.

### Milestone 3 — Test the most valuable data addition

Compare the full mixture with a matched-budget version in which the executable-policy slice is replaced by eligible public training examples. Keep initialization, approximate training tokens, and evaluation cases comparable. This tests whether changing-rule data adds transferable decision skill beyond more public-task training.

If the first pilot works, add the validated ShARC adapter and a small independent set from an intended application. Measure what changes before expanding teacher-generated probabilities or scaling the corpus substantially.

### Milestone 4 — Optional teacher-target experiment

Choose a bounded training subset. Obtain teacher distributions over exactly the same event definitions and candidates. Preserve original gold labels and compare gold-only against gold-plus-distillation at a matched budget. Grade on independent outcomes; teacher agreement is a secondary diagnostic.

Do not make RL a dependency of these milestones. These are supervised and distillation data preparations. A future RL experiment needs its own reward definition and outcome evaluation.

## 15. Completion criteria for the data-preparation build

- [ ] Source snapshots, published terms, and configurations are recorded; excluded/restricted sources are not silently included through aggregates.
- [ ] Each task specifies what its probabilities mean and which output type it uses.
- [ ] Direct adapters preserve relevant inputs, labels, and original split identities.
- [ ] Gold labels, exact posteriors, and teacher estimates have distinct provenance.
- [ ] Rules and priors necessary for inference are visible in the intended input.
- [ ] Root groups and known duplicate clusters do not cross train/evaluation boundaries.
- [ ] Model input excludes labels, explanations, hidden facts, and answer-derived annotations.
- [ ] No decisive content is silently truncated.
- [ ] Candidate changes have correct target remapping or relabeling.
- [ ] Schema/oracle checks pass, and systematic manual-audit findings have been resolved.
- [ ] Reports show actual canonical counts, root counts, augmentation counts, and rejected rows.
- [ ] Development, calibration, and final evaluation roles are frozen and documented.
- [ ] The data card limits claims to the tasks, languages, domains, and probability targets actually covered.

**Deliverable from implementing this plan:** a reproducible, versioned decision corpus and a defensible evaluation split—not a claim that 100,000 examples are sufficient for a universally calibrated model.
