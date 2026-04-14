# Evaluation Suite

Measures hybrid categorizer accuracy against hand-labeled ground truth.

## Run

```bash
uv run python -m eval.runner
```

Outputs `eval/latest_report.md` with overall accuracy + per-difficulty
+ per-source + failure breakdown + confusion pairs.

**This runs real LLM API calls.** Cost is ~$0.001 per 30 samples with
DeepSeek. Not part of CI, not run on every commit.

## Dataset schema (dataset.jsonl)

One JSON object per line:

- `description` (str, required) — merchant_clean text from Scotia CSV
- `sub_description` (str | null) — location/sub-detail, or null
- `expected_category` (str, required) — must be in `VALID_CATEGORIES`
- `difficulty` (str, required) — `easy` | `medium` | `hard`
- `notes` (str) — why this label, edge cases, rationale

## Difficulty guidelines

- **easy**: merchant name is obvious (`mcdonald's #8808` → fast_food)
- **medium**: needs mild reasoning (`arcteryx yorkdale` — brand+mall)
- **hard**: ambiguous, multi-category, or genuinely `uncategorized`
  (legal entity numbers, generic Square payments)

## Rules for good eval samples

- Pick merchants that **rules currently miss** — otherwise you're
  measuring the rule table, not the LLM
- Spread across difficulties — ~40% easy / 40% medium / 20% hard
  so you can see where the model breaks down
- Write notes in whichever language (中/英) keeps you precise