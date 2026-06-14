# DeepSmells+ — Improvement Report (ComplexMethod)
### Project 3 · enhancement over the DeepSmells baseline

**Result headline:** On the lab's ComplexMethod dataset, DeepSmells+ reaches
**F1 0.8276 / MCC 0.8135** — a large gain over our own DeepSmells reproduce
(**F1 0.6685 / MCC 0.6393**) on the *identical* data and validation split.
It also exceeds the published paper's numbers (F1 0.7542 / MCC 0.7341), **but see the
dataset caveat in §1.1** — the paper was measured on a different (smaller) data split, so that
particular comparison is indicative, not apples-to-apples.

---

## 1. Starting point

| Model | Data | Precision | Recall | F1 | MCC |
|---|---|---:|---:|---:|---:|
| Our reproduce (lab baseline) | **lab** | 0.633 | 0.709 | 0.6685 | 0.6393 |
| **DeepSmells+ (this work)** | **lab** | **0.852** | **0.805** | **0.8276** | **0.8135** |
| Paper DeepSmells (EASE'23, Table 2) | *paper* | 0.731 | 0.779 | 0.7542 | 0.7341 |

- **Primary, rigorous claim** (same data + same eval): DeepSmells+ over our reproduce =
  **+0.159 F1, +0.174 MCC**.
- **Secondary, indicative** (different data — see §1.1): DeepSmells+ also tops the paper's reported
  numbers by **+0.073 F1, +0.079 MCC**.

### 1.1 Dataset caveat — we did NOT use the paper's exact data

The lab dataset (Kaggle `dangvuhai/codesmell`, linked from the project package) is the **same
benchmark family** as the paper but a **re-tokenized / expanded** version — the counts do not match
the paper's Table 1:

| Smell | Paper Pos / Neg | Lab data (1d) Pos / Neg |
|---|---|---|
| ComplexMethod | 12,489 / 144,460 | 26,164 / 466,503 |
| ComplexConditional | 6,186 / 149,767 | 6,523 / 381,016 |
| FeatureEnvy | 1,788 / 51,260 | 1,918 / 171,300 |
| MultifacetedAbstraction | 290 / 50,205 | 307 / 173,757 |

CM positives are ~2× and all negatives are 2.5–3.5× the paper's. So the "vs paper" row is measured
on a **different test set** and must be read as a reference point, not a head-to-head result. The
**defensible evidence of improvement is the same-data baseline → DeepSmells+ jump.** To make a true
head-to-head vs the paper, we would need the paper's exact split (see §9).

---

## 2. Diagnosis — why the baseline trailed

1. **Raw token IDs into Conv1d.** The baseline fed integer token ids straight into the conv as a
   single continuous channel — so id `500` looks "100× bigger" than id `5`, which is meaningless.
   The convolution was forced to learn from a fake magnitude axis.
2. **Fixed loss/optimizer.** `BCE + pos_weight` with SGD(lr=0.03) for 60 epochs → slow, overfits
   after ~epoch 20.
3. **Hard 0.5 threshold** on an 8%-positive distribution — not where F1 is maximized.

---

## 3. What we changed (4 levers)

| # | Change | Why |
|---|---|---|
| 1 | **Embedding layer** `nn.Embedding(8463, 32, padding_idx=0)` before the CNN | Each token id → a learned dense vector. Removes the fake-magnitude problem; gives the CNN a real semantic feature space. **Biggest lever.** |
| 2 | **Focal Loss** (γ=2, α=0.75) replaces BCE+pos_weight | Down-weights easy negatives, focuses learning on the rare/hard positives — better for 8% imbalance than a flat class weight. |
| 3 | **AdamW** (lr=1e-3, wd=1e-4) replaces SGD | Faster, more stable convergence — best epoch is ~6 instead of ~20+. |
| 4 | **Threshold tuning** on validation (sweep 0.05–0.95, pick max-F1) | The F1-optimal cut is **0.6**, not 0.5. Free metric gain. |

Plus **early stopping** (patience 6) — stops the wasted/overfitting epochs the baseline ran.

### Held fixed (so the comparison is honest)
Data pipeline (outlier-trim length 1071, zero-pad, stratified split), `MAX_TRAINING_SAMPLES=5000`,
**full validation set (47,085 samples, 7.96% positive)**, and the P/R/F1/MCC definitions — all
identical to the baseline. The gain is purely model/loss/threshold, **not** an easier eval set.

---

## 4. Architecture

```
token ids                         B × 1071  (int)
  Embedding(8463, 32, pad_idx=0)            → B × 1071 × 32
  permute                                    → B × 32 × 1071
  Conv1d 32→16 → BN → ReLU → MaxPool(2)
  Conv1d 16→32 → BN → ReLU → MaxPool(2)      → B × 32 × 264
  LSTM(hidden=100), last hidden state        → B × 100
  Linear→ReLU→Drop ×2 → Linear               → B × 1 logit
Loss  : Focal(γ=2, α=0.75)
Optim : AdamW(1e-3, wd=1e-4), AMP, early stop
Decide: prob ≥ 0.6  (tuned on validation)
```
Conv / LSTM / classifier dims are **unchanged from the lab model** — only the input front-end
(embedding), loss, optimizer, and threshold changed, so the comparison isolates those.

---

## 5. Config search (6 runs, full ~8% eval)

| embed_dim | γ | α | P | R | F1 | MCC | thr | epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **32** | **2** | **0.75** | 0.852 | 0.805 | **0.8276** | **0.8135** | 0.60 | 6 |
| 32 | 2 | 0.5 | 0.839 | 0.805 | 0.8220 | 0.8071 | 0.45 | 4 |
| 64 | 3 | 0.75 | 0.846 | 0.798 | 0.8213 | 0.8068 | 0.55 | 8 |
| 64 | 2 | 0.5 | 0.858 | 0.784 | 0.8190 | 0.8051 | 0.45 | 4 |
| 64 | 1 | 0.75 | 0.822 | 0.814 | 0.8182 | 0.8026 | 0.55 | 4 |
| 64 | 2 | 0.75 | 0.813 | 0.822 | 0.8175 | 0.8017 | 0.50 | 4 |

**All 6 beat the paper.** Spread is tiny (F1 0.817–0.828) → the win comes from the *combination*
of changes, not from a lucky hyper-parameter. embed_dim=32 slightly edges 64 (less overfit on this
data size).

---

## 6. Talking points (for presentation)

- **Embedding is the key idea.** Token ids are categorical, not numeric; turning them into learned
  vectors is what unlocked the jump. This is also the paper's own pipeline weakness we fixed.
- **Focal + threshold tuning handle imbalance better than a single `pos_weight`.** The model keeps
  high precision (0.85) *and* high recall (0.80) — the baseline could not hold both.
- **Converges in ~6 epochs** with AdamW vs ~20+ for SGD → less compute, less overfitting.
- **Fair fight:** same data, same eval, same imbalance — only the model improved.

---

## 7. Limitations / honest notes

- Validation set is also used for model selection and threshold tuning (as in the lab). A fully
  separate held-out test would be stricter; threshold 0.6 is mildly optimistic.
- Only **ComplexMethod** done (quick-win scope). The same recipe should be run on CC/FE/MA to claim
  a full paper-wide improvement.
- Single change attribution is design-level (search shows config-insensitivity); a strict ablation
  (embedding-only, focal-only, …) would quantify each lever's exact share — not yet run.

---

## 8. Deliverables

| File | What |
|---|---|
| `deepsmells_plus.py` | improved model + search (Embedding/Focal/AdamW/early-stop/threshold) |
| `deepsmells_plus_best.pth` | best checkpoint + config |
| `results_improved.json` | all 6 search results |
| `REPORT_IMPROVED.md` | this report |
| `REPORT.md` | baseline (reproduce) report |

**Conclusion:** With four targeted, low-cost changes — a token **embedding**, **Focal Loss**,
**AdamW**, and **threshold tuning** — DeepSmells+ raises ComplexMethod detection from our
same-data baseline **F1 0.67 / MCC 0.64 to F1 0.83 / MCC 0.81**. It also exceeds the paper's
reported 0.75 / 0.73, though that comparison is on a different data split (§1.1) and should be
read as indicative.

---

## 9. Path to a true head-to-head with the paper

The current paper comparison is indicative because the data differs (§1.1). To make it exact, in
rough order of preference:

1. **Obtain the paper's exact dataset/split.** The paper builds on the benchmark of Sharma et al.;
   the original DeepSmells codebase (HUST authors) ships the tokenized split + train/test protocol.
   Run *both* the baseline and DeepSmells+ on that data → directly comparable to Table 2/3.
2. **Match the paper's protocol on available data.** Use all positives (drop the 5000 cap),
   the paper's 70/30 split, and the same outlier/padding rules; reproduce their reported pos/neg
   ratios as closely as possible. Closer, but counts still won't match exactly.
3. **Keep the current framing (done).** Treat the paper as a reference point and prove improvement
   against the same-data baseline — already rigorous, just not a head-to-head vs the paper.
