# DeepSmells-CodeBERT — Raw-source code smell detection

**Date:** 2026-06-14 · **Branch:** `feat/raw-source-detection`
**Strategy:** Fine-tune a pretrained code language model (**CodeBERT**) on **raw source code**
(`.code` files) instead of the paper's token-index → 1D-CNN. Covers **C# + Java**, **4 smells**.

## Motivation
The paper explicitly avoids learned embeddings ("…our work encodes the whole source code by
tokenization indexing…"), feeding bare integer token indices to a 1D-CNN. CodeBERT brings real
sub-word code embeddings + pretrained transformer context — the principled modern upgrade.

## Data
Source: Tushar Sharma `DeepLearningSmells/data/training_data_{cs,java}` — raw `.code` method/class
fragments, byte-identical origin to the lab's tokenized data. 8 archives (4 smells × 2 langs).

Raw counts (positives are scarce; negatives shared across smells, huge):

| lang | smell | pos | neg |
|---|---|---:|---:|
| cs | ComplexMethod | 25,867 | 921,035 |
| cs | ComplexConditional | 6,089 | 940,813 |
| cs | FeatureEnvy | 2,187 | 302,887 |
| cs | MultifacetedAbstraction | 386 | 304,688 |
| java | ComplexMethod | 45,273 | 2,226,380 |
| java | ComplexConditional | 20,379 | 2,251,274 |
| java | FeatureEnvy | 3,429 | 315,147 |
| java | MultifacetedAbstraction | 847 | 317,729 |

## Sampling (tractable fine-tuning)
Per (lang, smell): `pos_keep = min(pos_all, 8000)`, `neg_keep = min(5·pos_keep, neg_all)`
(≈1:5 → ~17% positive). Negatives sampled with a fixed seed. Max ~48k samples/combo.
Build a compact `data_raw/<lang>_<smell>.jsonl` of `{code, label}` by **targeted extraction** from
the `.7z` (no mass file extraction), then delete temp.

## Model & training
- `microsoft/codebert-base` + sequence-classification head (2 logits), `max_len=512` (long
  methods truncated — a known CodeBERT limitation, documented).
- Fine-tune 3 epochs, AdamW lr=2e-5, fp16, batch 16, class-weighted cross-entropy for residual
  imbalance.
- Per (lang, smell) → one binary classifier (**8 models**).
- Stratified 70/30 split; metrics **P / R / F1 / MCC**; tune decision threshold on validation.

## Evaluation caveat
Validation uses the **sampled 1:5 distribution (~17% positive)**, not the true <3% imbalance. So
absolute F1 is **not directly comparable to the paper's true-imbalance Table 2/3**. The comparison
here is: (a) CodeBERT raw-source vs our token-index baseline, and (b) cross-language behaviour.
A true-imbalance eval would need inference over millions of negatives (out of scope).

## Compute
Extraction ~20 min total; fine-tuning 8 combos × 3 epochs ≈ 2–3 h on RTX 5060 Ti (fp16).
First run downloads CodeBERT (~500 MB).

## Data cleanup (this branch)
Remove: extracted `data/` (2 GB, reproducible from `archive.zip`), `sharma_cs/` temp `.7z`.
Keep: `archive.zip`, `raw/*.7z` (new source), reports, prior notebooks/checkpoints.

## Deliverables
- `build_dataset.py` — archive → sampled JSONL.
- `codebert_smells.py` — fine-tune + eval all 8 combos.
- `results_codebert.json`, best checkpoints.
- `REPORT_RAW.md` — approach, 8-combo results, vs token-baseline, caveats.

## Success criteria
- Pipeline runs end-to-end on all 8 combos, produces P/R/F1/MCC.
- CodeBERT clearly beats the token-index baseline's discrimination on matched combos.
- Documented cross-language (cs vs java) comparison.
