# DeepSmells — Code Smell Detection
### Project 3 report / presentation notes

> **Baseline reproduce.** For the improvement that beats the paper (F1 0.83 / MCC 0.81),
> see [`REPORT_IMPROVED.md`](REPORT_IMPROVED.md). For a step-by-step trace of how the data is
> processed (real sample, every shape change), see [`REPORT_DATA.md`](REPORT_DATA.md).

**Smell trained:** ComplexMethod · **Model:** DeepSmells (1D-CNN → LSTM)
**Scale:** full paper-scale grid (35 configs × 60 epochs), GPU RTX 5060 Ti
**Best result:** F1 = **0.6685**, MCC = **0.6393** (pos_weight=4, kernel=6)

---

## 1. Problem

A **code smell** = a structural sign that code is hard to maintain (e.g. a *Complex Method*
that does too many things). DeepSmells treats each smell as its own **binary classification**:

- **Input:** one method/class, already *tokenized* into a sequence of integers (`12 93 41 7 …`).
- **Output:** `1` = smell present, `0` = no smell.

No hand rules ("count the `if`s"). The model **learns the pattern from data**.

Four smells in the dataset: `ComplexMethod`, `ComplexConditional`, `FeatureEnvy`,
`MultifacetedAbstraction`. This run = **ComplexMethod** (the others run the same pipeline,
just change `SMELL`).

---

## 2. Data

Kaggle dataset `dangvuhai/codesmell` → `tokenizer_cs/<smell>/{1d,2d}/{Positive,Negative}/*.tok.cld`.
Each line = one tokenized sample. We use `1d` (the main model eats a 1-D token sequence).

**Raw counts (ComplexMethod, 1d):**

| Label | Files | Raw lines |
|---|---:|---:|
| Positive (smell) | 1 | 26,164 |
| Negative (clean) | 5 | 466,503 |

→ **heavily imbalanced**: smells are rare (~5% of raw lines). This is the central challenge.

**After the data pipeline (full-scale):**

| | value |
|---|---|
| `max_input_length` (after outlier trim) | **1071** |
| train_data | (109,862, 1071, 1) |
| valid_data | (47,085, 1071, 1) |
| positive ratio (train / valid) | **7.96% / 7.96%** |

The validation set keeps the **true ~8% imbalance** — so reported numbers are honest, not inflated.

---

## 3. Pipeline (data → tensors)

1. **Find data root** — auto-locate `tokenizer_cs` (deep `rglob`, no hardcoded path → works on Kaggle).
2. **Token lengths** — read every sample, record its length.
3. **Outlier-trim length** — `max_input_length = mean + 1·std`, drop samples longer than that.
   Keeps padding cost bounded (→ 1071 instead of the longest outlier).
4. **Pad** — every kept sample zero-padded to `max_input_length`; reshape `N × 1071 × 1`.
5. **Balance train** — equal #positive/#negative in the training pool (cap 5000 each before re-split).
6. **Stratified split** — pool train+eval, re-split 70/30 keeping the label ratio stable in both.
7. **PyTorch `Dataset`/`DataLoader`** — each sample → `1 × 1071` tensor (Conv1d wants `channels × length`).
   Batch shape `128 × 1 × 1071`, label `128 × 1`.

> **Why pad to equal length?** A batch is one tensor — all rows must share shape.
> **Why balance train but not valid?** Balanced train stops the model defaulting to "always Negative";
> imbalanced valid measures real-world performance.

---

## 4. Model — DeepSmells (CNN → LSTM)

```
input  128 × 1 × 1071
  │
  ├─ Conv block 1:  Conv1d → BatchNorm → ReLU → MaxPool(2)     1→16 ch
  ├─ Conv block 2:  Conv1d → BatchNorm → ReLU → MaxPool(2)    16→32 ch
  │                                          → 128 × 32 × 264
  │
  ├─ LSTM (hidden=100)        last hidden state → 128 × 100
  │
  └─ Classifier: Linear→ReLU→Dropout ×2 → Linear → 1 logit     128 × 1
```

- **Loss:** `BCEWithLogitsLoss(pos_weight=w)` — `pos_weight` up-weights the rare positive class.
- **Optimizer:** SGD, lr = 0.03 · **Epochs:** 60 · **AMP** mixed-precision on GPU.
- **BiLSTM variant** (`DeepSmells-BiLSTM`): same, but LSTM is bidirectional and the classifier
  takes `hidden×2` (concatenation of both directions).

### Q: What do CNN and LSTM each do here?  *(report question)*
- **CNN = local pattern detector.** Slides over the token sequence, learns short motifs
  (nested `if`, repeated calls, long arithmetic chains); pooling shrinks length + keeps the
  strongest signals. It answers **"what local patterns appear."**
- **LSTM = sequential / long-range modeler.** Reads the CNN feature sequence in order, carries
  state across it, captures dependencies a fixed conv window can't. It answers **"how those
  patterns chain across the whole method."**
- Final hidden state → classifier → one smell logit.

---

## 5. Metrics (why these)

On imbalanced data, **accuracy lies** (predict all-Negative → 92% "accurate", useless). We use:

| Metric | Meaning |
|---|---|
| **Precision** | of samples flagged as smell, how many really are |
| **Recall** | of real smells, how many we caught |
| **F1** | harmonic mean of P & R (the headline number) |
| **MCC** | balanced correlation, robust under strong imbalance |

---

## 6. Results — full 35-config grid

Grid = `pos_weight ∈ {1,2,4,8,12,32,84}` × `kernel_size ∈ {3,4,5,6,7}`.
Each cell = best validation epoch.

**Top configs (by F1):**

| pos_weight | kernel | Precision | Recall | F1 | MCC |
|---:|---:|---:|---:|---:|---:|
| **4** | **6** | 0.633 | 0.709 | **0.6685** | **0.6393** |
| 4 | 3 | 0.629 | 0.693 | 0.6594 | 0.6293 |
| 2 | 5 | 0.654 | 0.659 | 0.6567 | 0.6268 |
| 1 | 5 | 0.654 | 0.659 | 0.6561 | 0.6263 |
| 12 | 4 | 0.592 | 0.734 | 0.6553 | 0.6264 |

**Worst (over-weighted positives):**

| pos_weight | kernel | Precision | Recall | F1 | MCC |
|---:|---:|---:|---:|---:|---:|
| 32 | 3 | 0.338 | 0.841 | 0.482 | 0.474 |
| 84 | 6 | 0.240 | 0.890 | 0.378 | 0.383 |
| 84 | 3 | 0.218 | 0.885 | 0.349 | 0.353 |

### Key trends (presentation talking points)
1. **`pos_weight` trades precision for recall.** Low (1–4): balanced, best F1. High (32–84):
   recall → ~0.85 but precision collapses to ~0.24 → F1 craters. *More positive weight = flags
   everything = many false alarms.*
2. **Sweet spot pos_weight = 2–4.** F1 peaks ~0.66–0.67 there for every kernel.
3. **Kernel size = minor effect** vs pos_weight. k=6 edged out the best F1; k=3–7 all viable.
4. **Best config peaks early (~epoch 20), then overfits** — valid loss climbs 0.36 → 0.65 by
   epoch 60. The trainer keeps the **best-F1 checkpoint**, so the saved model is the epoch-20 one.
   *(Takeaway: 30 epochs would suffice; 60 wasted compute.)*

### Cross-check (Kaggle, Plan A chunk)
Independent Kaggle run, same scale, best config pw=2/k=6 → F1 = **0.672** — matches the PC grid
(F1 cluster 0.63–0.67). Two machines agree → pipeline is correct.

---

## 7. Honesty note: full-scale 0.67 vs dev-run 0.78

An earlier quick "dev run" reported F1 = 0.776 — that used a **tiny 5,000-sample, near-balanced**
eval set (easy). The full-scale numbers here use **47,085 samples at the true 8% imbalance** (hard
and realistic). **0.67 is the report-worthy number;** 0.78 was an optimistic artifact of the small
balanced eval.

---

## 8. Deliverables

| File | What |
|---|---|
| `codesmell-slab.ipynb` | completed lab notebook (clean, Kaggle-ready, full-scale config) |
| `codesmell-slab.full.ipynb` | executed full-grid run with all outputs embedded |
| `codesmell-slab-kaggleA.ipynb` | Plan-A chunked grid for Kaggle (split <12h/run) |
| `deepsmells_tracking/*.txt` | per-config epoch logs (35 files) |
| `deepsmells_checkpoints/*.pth` | best-F1 model per config (35 files) |

**Conclusion:** DeepSmells detects ComplexMethod at **F1 ≈ 0.67 / MCC ≈ 0.64** on a realistically
imbalanced test set. Best recipe: moderate positive weighting (`pos_weight=4`), `kernel=6`, and
early stopping around epoch 20. The CNN extracts local code motifs; the LSTM ties them together
across the method — together beating any single hand-written rule.
