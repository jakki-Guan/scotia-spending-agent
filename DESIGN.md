# Design Decisions

> A companion to `README.md`. The README tells you *what* this project is.
> This document tells you *why* it's shaped the way it is, what was tried,
> and what was deliberately left out.
>
> Last substantive update: 2026-04-12

---

## 1. Problem framing

### Why an agent, not a report generator

A static "categorize and chart" pipeline would have been faster to build and
would have looked fine in a README GIF. It would also have been a less
honest portfolio piece — nothing in it demonstrates the ability to work
with LLMs as reasoning components rather than as text-completion APIs.

So the core loop is a tool-calling agent: the LLM is handed a small set of
analysis tools, decides which to invoke based on the user's question,
interprets their output, and composes an answer. Every tool call is visible
in the UI. The reader is meant to be able to watch the agent think.

### Why hybrid categorization, not pure LLM

Classifying every transaction through an LLM is wasteful and slow.
Classifying with rules alone breaks on the long tail (one-off merchants,
legal-entity strings, bilingual/romanized names). The hybrid layer splits
the work along the distribution:

- **Head — keyword rules.** Fast, free, deterministic, easy to test.
  A single rule like `("tim hortons", "coffee")` covers hundreds of real
  rows in the dataset.
- **Tail — LLM.** Handles whatever rules miss, including genuinely
  ambiguous cases where the right answer is "I don't know."

This split produces a system whose cost scales with the *novelty* of the
input, not its volume.

---

## 2. Architecture at a glance

```
CSV file
   │
   ▼
parser.py          →  (DataFrame, [ValidationError])   Pydantic per-row
   │
   ▼
hybrid.py          →  CategoryResult (src=rule|llm|fallback)
   │                          │
   │                          ├─ categories.categorize   (keyword rules)
   │                          └─ llm_categorize          (DeepSeek API)
   ▼
visualizer.py      →  charts (Phase 1, done)
   │
   ▼
agent.py           →  tool-calling loop (Phase 2, in progress)
   │
   ▼
Gradio UI          →  chat + charts + tool-call transparency (Phase 3)
```

The data contract between stages is deliberately narrow: each stage
consumes and produces one well-typed object. Nothing reaches across
layers.

---

## 3. Hybrid categorization

### 3.1 Miss-only fallback, not confidence-gated

`hybrid_categorize` dispatches in one direction:

```python
rule_cat = categorize(description)
if rule_cat != "uncategorized":
    return CategoryResult(..., confidence=1.0, source="rule")
return llm_categorize(description, sub_description)
```

The alternative — "let rules run, but send low-confidence rule hits to the
LLM for a second opinion" — was considered and rejected. The rule layer
has no native confidence signal; it fires or it doesn't. Manufacturing a
synthetic confidence threshold would layer a new invariant on top of the
already-tested rule-ordering invariant, and it would make "why did this
transaction get re-classified?" harder to reason about.

Keeping rules as a **hard contract** (fire → trust → confidence=1.0) makes
the system easier to debug and easier to test. The cost is that a
mis-written rule will silently produce wrong answers; the mitigation is
that rules are covered by unit tests and are short enough to eyeball.

### 3.2 Merchant aliasing inside the rule table

Not every rule is a naïve "keyword in name → category" mapping. Several
entries are the result of manual investigation into opaque strings that
appear in Scotia's CSV:

| Raw description              | Rule category | Story                                  |
|------------------------------|---------------|----------------------------------------|
| `1000503499 ontario limit`   | `fuel`        | Legal entity for an Esso gas station   |
| `parchment`                  | `education`   | Credential-verification service        |
| `vue*testing`                | `education`   | Online proctoring platform             |
| `paymentus`                  | `utilities`   | Utility payment processor              |

Each of these was unfindable from the string alone. The fix is not a
smarter matcher — it's persistent detective work, cached as a rule.
This aliasing layer is a *feature* of the categorizer, not a workaround,
and the eval harness has an `aliased: bool` field to measure its
contribution separately from LLM inference (§5.2).

### 3.3 Honest "uncategorized" as a first-class output

A classifier that always picks *some* category is a classifier that lies
on inputs it can't interpret. The taxonomy explicitly includes
`uncategorized` as a terminal bucket, and both layers can return it:

- **Rules** return `uncategorized` when no substring matches.
- **LLM** returns it when the merchant string carries no semantic content,
  typically with a low confidence score. Example from the eval set:

  ```
  sq *ahc brothers inc  →  uncategorized  (conf=0.20)
  ```

  A Square payment to a generic legal entity — no inferrable category.
  The LLM produces the right answer *and* flags its low confidence,
  which is exactly the behavior a downstream UI needs to show an
  uncertainty indicator.

A classifier that says "I don't know, and I'm not sure I don't know"
is more useful than one that always commits.

---

## 4. Defensive parsing — the rescue path

### 4.1 What happens in the wild

DeepSeek's JSON mode is good but not perfect. Roughly 1–2% of responses
arrive with JSON defects — most commonly an unescaped quote inside a
string value:

```
{"category": "restaurant", "confidence": 0.95,
 "reasoning": "a "hotpot" place"}
```

`json.loads` rejects this. Before the rescue path existed, the entire
response was discarded and the caller degraded to `source="fallback"`,
losing information the LLM had correctly produced.

### 4.2 Two-stage parsing

`_parse_response` tries strict JSON first, then falls back to per-field
regex rescue:

| Stage | Behavior on success                         | Behavior on failure                 |
|-------|---------------------------------------------|-------------------------------------|
| 1     | Full structured `CategoryResult`            | Catch `JSONDecodeError`, try Stage 2|
| 2     | Regex-extract `category` (required),        | Raise `ValueError` — caller degrades|
|       | `confidence` & `reasoning` (best-effort)    | to `source="fallback"`              |

Stage 2 design notes:

- **Category is required**; if even regex can't find it, the response is
  unusable and we give up honestly.
- **Confidence is reported honestly when regex can extract it.** An earlier
  version of Stage 2 always set confidence to 0.5 ("penalty for rescue").
  That discarded real signal — if the LLM wrote `"confidence": 0.95` in
  a response that happened to have a malformed reasoning field, our
  penalty was punishing the wrong field. The current version salvages
  each field independently.
- **Reasoning is prefixed with a marker**: `[rescued from malformed JSON]`.
  Downstream eval and UI can distinguish clean from rescued results by
  substring check on `reasoning`, without needing a new schema field.

### 4.3 Why a marker string, not a new `source` value

`CategoryResult.source` is a `Literal["rule", "llm", "fallback"]`. Adding
`"llm_rescued"` was considered and rejected:

- The `source` field answers "which layer produced this?" — rescue happens
  inside the LLM layer, not outside it. A rescued result is still an
  LLM result.
- Adding a fourth literal forces every downstream switch/match to grow a
  branch, including places that legitimately don't care about rescue
  (e.g., the Gradio UI's "🔧 tool call" indicator).
- A marker string in `reasoning` is filterable (`startswith("[rescued")`)
  without schema churn.

The rule is: new schema fields when the information is *structural*,
marker conventions when it's *forensic*.

---

## 5. Evaluation methodology

### 5.1 Why a hand-labeled set

Rule coverage on real data is easy to measure (row hit rate, dollar hit
rate). LLM behavior on the long tail is not. The eval set gives a fixed
30-item yardstick that can be re-run after any prompt change, model
swap, or rule-table update.

Current composition (31 samples):

- 6 rule-layer **smoke tests** (incl. merchant aliases like `1000503499`)
- 25 rule-miss samples drawn from 57 unique uncategorized merchants
  observed in real Scotia data

Classes covered: `bubble_tea`, `dessert`, `fast_food`, `restaurant`,
`groceries`, `pharmacy`, `parking`, `fuel`, `government_fees`,
`education`, `shopping_retail`, `coffee`, `uncategorized`.

### 5.2 Schema choices

JSONL, one sample per line:

```jsonc
{
  "description": "...",          // merchant_clean from the CSV
  "sub_description": null,       // location/detail, if present
  "expected_category": "...",    // must be in VALID_CATEGORIES
  "difficulty": "easy|medium|hard",
  "aliased": false,              // was this covered by a rule alias?
  "notes": "..."                 // rationale, in whatever language fits
}
```

Two non-obvious fields:

**`difficulty`** describes how hard the string *itself* is to classify
from inference alone — it's independent of whether the current system
happens to get it right. An easy sample (`mcdonald's #8808`) can still
be answered by the LLM incorrectly; a hard sample (`1000503499 ontario
limit`) might be correctly answered because a rule aliases it. Keeping
difficulty decoupled from system capability lets the eval report a
2-D breakdown (per-source × per-difficulty) that tells a richer story
than overall accuracy.

**`aliased`** separates two distinct questions the eval wants to answer:

1. *Is the system currently correct on this row?* — main accuracy metric.
2. *Could the LLM alone have gotten this right?* — diagnostic metric.

Without `aliased`, a merchant like `1000503499 ontario limit` would
either inflate LLM-credit (if classed as an LLM win) or be confusingly
excluded. With it, both questions are answerable from the same dataset.

### 5.3 Labeling philosophy: business format, not customer intent

Two adjacent categories — `fast_food` and `dessert` — caused the sharpest
labeling ambiguity. Resolved with a written rule:

> Classify by **operational business format** (quick-serve kiosk, sit-down
> restaurant, dedicated dessert shop), not by **customer motivation** or
> menu breadth.

Worked examples from the eval set:

- `cinnabon` → `dessert` (a dedicated sweets shop that happens to be a chain)
- `dairy queen` → `fast_food` (a quick-serve chain that also sells ice cream)
- `mr. pretzels` → `fast_food` (by analogy to the DQ case — mall kiosk with savory primary product)
- `pizza bell and wings` → `fast_food` (revised from `restaurant` during eval — see §5.5)

### 5.4 Current performance

From `uv run python -m eval.runner`:

| Metric        | Value        |
|---------------|--------------|
| **Overall**   | **29/31 = 93.5%** |
| Rule layer    | 6/6 = 100%   |
| LLM layer     | 23/25 = 92%  |

Confusion pairs on the failures: `fast_food → dessert`, x2.

### 5.5 Two known failures, kept on purpose

The eval set has two samples the LLM consistently mislabels:

| Sample              | Expected    | LLM says | Why LLM is wrong (under our taxonomy) |
|---------------------|-------------|----------|---------------------------------------|
| `dairy queen #12338`| `fast_food` | `dessert`| DQ's primary line is fast food        |
| `mr. pretzels`      | `fast_food` | `dessert`| Savory kiosk snack, same pattern as DQ|

These *could* be fixed by relabeling to `dessert`, which would push the
eval score to 31/31. That was considered and rejected: both cases sit on
a genuinely fuzzy taxonomy boundary (mall kiosk + snack format), and a
100% eval set is a set that has stopped teaching anything. Keeping them
as documented failures preserves the signal.

A third sample — `pizza bell and wings` — *was* relabeled mid-eval (from
`restaurant` to `fast_food`), because the original label was inconsistent
with the business-format principle in §5.3. This is recorded in the
sample's notes with a `REVISED:` prefix, so the labeling history is
visible in the JSONL file itself.

### 5.6 What the eval does not cover

- **Only 1 CSV's worth of merchants** (~1000 rows, 267 uniques). Long-tail
  behavior on merchants not present in this data is untested.
- **Only one LLM provider** (DeepSeek via OpenAI-compatible API).
  Provider-specific failure modes (e.g. the JSON-key bug in §4) may
  differ on Anthropic / Ollama backends.
- **No adversarial inputs.** Prompt injection through merchant names
  (unlikely but possible) is not tested.

---

## 6. Testing strategy

Three layers, each with a deliberately different scope:

### 6.1 Unit tests (`tests/`) — run on every commit

- `test_parser.py` — per-row Pydantic validation, BOM handling, sign
  convention preservation.
- `test_categories.py` — rule hit behavior, ordering invariant
  (`ubereats` before `uber`), dual dollar/row metric.
- `test_llm_categorize.py` — **pure-function** tests on `_parse_response`,
  including a regression suite for the malformed-JSON rescue path. No
  network, no mocks; ~17 tests, millisecond runtime.
- `test_hybrid.py` — dispatch-layer tests with `unittest.mock` patching
  `scotia_agent.hybrid.llm_categorize` (use-site binding, not
  definition-site). Confirms rules-hit skips the LLM and rule-miss
  forwards arguments correctly.

### 6.2 Eval suite (`eval/`) — run on demand, not in CI

Makes real LLM API calls. Cost is small (~$0.001/run with DeepSeek)
but non-zero and non-deterministic — unsuitable for per-commit CI.
Kept deliberately outside the `tests/` tree so pytest doesn't discover
it by accident.

### 6.3 Integration (planned, Phase 4)

Full parser → hybrid → visualizer run on the real CSV, producing the
same charts the Gradio app produces. Asserts row/$ hit rates per source
stay above floors (e.g., "rules must cover ≥90% of dollars").

### 6.4 Mock style: patch where used

`test_hybrid.py` mocks `scotia_agent.hybrid.llm_categorize` — the name
as bound *inside* `hybrid.py` — rather than the name in `llm_categorize`
where it is defined. This is the standard Python monkey-patch pattern;
patching the definition site would not intercept the dispatcher's call
because `from ... import llm_categorize` creates a fresh reference in
the importing module.

---

## 7. Known limitations and future work

### Now

- **LLM non-determinism.** A DQ classification might come back `fast_food`
  one run and `dessert` the next. Eval results are reported from a
  single run and can fluctuate by ±1 on a 31-sample set.
- **Eval set is small (31).** Statistically, 93.5% ± a few percent.
- **No caching of LLM calls.** A repeat run of the eval pays for every
  merchant again. Fine at 31 samples, painful at 1000.

### Phase 3 (next)

- Gradio UI with tool-call transparency (visible "🔧 calling
  get_spending_by_category(month='2025-03')" during agent reasoning).
- File upload flow; drag-and-drop a Scotia CSV, see charts + chat.

### Phase 4

- Integration test on real CSV with hit-rate floors.
- Anonymized sample CSV for public demo.
- Deploy to Hugging Face Spaces.

### Not planned

- **Embedding-based categorization** (ChromaDB + sentence-transformers).
  Reserved as a future extension, but keyword rules + LLM fallback
  already cover the dataset at 93.5%. Adding a third layer would need
  a clear failure mode it uniquely solves, and one hasn't appeared yet.
- **Multiple LLM providers in production.** The abstraction exists
  (OpenAI-compatible client), but the project will ship with DeepSeek
  primary + Ollama fallback only. Multi-provider orchestration is not
  a portfolio-differentiating feature for this role target.
- **User accounts / persistence.** This is a single-user tool analyzing
  one person's CSV. Adding auth would be yak-shaving.