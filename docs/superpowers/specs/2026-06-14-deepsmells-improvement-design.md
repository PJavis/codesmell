# DeepSmells+ — Improvement over baseline (ComplexMethod)

**Date:** 2026-06-14
**Scope:** Quick-win improvements on the ComplexMethod (CM) smell only.
**Goal:** Beat the paper's DeepSmells CM result — **F1 = 0.7542, MCC = 0.7341** (Table 2, EASE'23).
Our faithful reproduce got F1 = 0.6685 / MCC = 0.6393, so we are improving from that floor.

## Why the reproduce trails the paper (diagnosis)

1. **Training positives capped at 5,000** (`MAX_TRAINING_SAMPLES=5000`); the paper uses all CM
   positives. Removing the cap is a free, large lever.
2. **Raw token IDs fed straight into Conv1d** — treats id 500 as numerically "bigger" than id 5,
   which is semantically meaningless. An embedding layer is the principled fix and expected to be
   the single biggest gain.
3. **SGD lr=0.03 for 60 epochs** — slow convergence and overfitting after ~epoch 20
   (valid loss climbs 0.36 → 0.65).
4. **Fixed decision threshold 0.5** on an 8%-positive distribution — not F1-optimal.

## Design — DeepSmells+ (minimal-invasive enhancement)

Keep the entire data pipeline (outlier-trim length=1071, zero-pad, stratified 70/30 split, true
~8% imbalance in validation) and the metric definitions (P/R/F1/MCC). Change only the model head,
loss, optimizer, training control, and decision threshold.

### Architecture
```
token ids                       B × 1071  (int64)
  Embedding(num_embeddings=8463, dim=32, padding_idx=0)   → B × 1071 × 32
  permute                                                  → B × 32 × 1071   (32 channels)
  Conv block 1: Conv1d→BatchNorm→ReLU→MaxPool(2)   32→16 ch (keep paper conv dims)
  Conv block 2: Conv1d→BatchNorm→ReLU→MaxPool(2)   16→32 ch  → B × 32 × L'
  LSTM(input_size=L', hidden=100), take last hidden state    → B × 100
  Classifier: Linear→ReLU→Dropout ×2 → Linear              → B × 1 logit
```
- Token vocab measured from data: ids 33..8462 → `num_embeddings = 8463`; `padding_idx=0` (pad token
  0 sits below the real-token range, so it is safe and learns a zero-ish vector).
- Embedding output (32) becomes the Conv1d input channels, replacing the single raw-magnitude channel.
- Conv/LSTM/classifier dims unchanged from the lab model so the comparison isolates the new pieces.

### Loss
- **Focal Loss** (binary) with `alpha`, `gamma` to focus on hard/rare positives, replacing
  `BCEWithLogitsLoss(pos_weight)`. Default start `gamma=2.0`, `alpha=0.75`.
- Keep a `BCE + pos_weight` fallback path for an apples-to-apples ablation if Focal underperforms.

### Optimizer & training control
- **AdamW**, lr=1e-3, weight_decay=1e-4 (replaces SGD lr=0.03).
- **Remove the positive training cap** (use all available positives; keep balancing of the train
  pool by undersampling negatives to match, as the lab does).
- **Early stopping** on validation F1, patience=8, max 60 epochs; keep the best-F1 checkpoint.
- AMP mixed precision on GPU (as before).

### Decision threshold
- After training, **sweep threshold** over validation probabilities and pick the F1-maximizing
  value; report that threshold alongside metrics (do NOT tune on a separate test — this is the
  validation set used for model selection, consistent with the lab).

### Config search (small)
- Embedding dim ∈ {32, 64}; Focal `gamma` ∈ {1.0, 2.0}; `alpha` ∈ {0.5, 0.75}.
- A handful of runs (≤8), not the full 35-cell grid. Pick best validation F1.

## What stays fixed (so the comparison is fair)
- Data loader, outlier trim (1071), padding, stratified split, seeds.
- Validation keeps the real imbalance (~8% positive).
- Metrics: Precision, Recall, F1, MCC.
- Smell = ComplexMethod, dim = 1d.

## Deliverables
- `codesmell-improved.ipynb` — self-contained improved notebook (Kaggle-ready, same data-finder).
- Comparison table: **reproduce (0.669)** vs **paper DeepSmells (0.754)** vs **DeepSmells+ (new)**.
- Updated `REPORT.md` with an "Improvement" section (what changed, why, results, ablation).

## Success criteria
- Primary: validation **F1 > 0.7542** and **MCC > 0.7341** on CM (beat paper).
- Secondary (if primary not reached): clearly beat the reproduce floor (F1 0.669) and document
  which single change contributed most (ablation: +embedding, +focal, +AdamW, +threshold).

## Compute / time
- ~half a day to build; one PC run (RTX 5060 Ti), ~3–6h for the small search.
- No Kaggle 12h concern (PC, no session limit).

## Risks
- Embedding + full positives raises memory; mitigate with batch size and `padding_idx`.
- Focal Loss can be unstable; AdamW + grad scaling and the BCE fallback mitigate.
- If still below paper, likely dataset-version differences (our CM has 26,164 positive *lines* vs
  paper's 12,489 *instances*); document rather than chase.
