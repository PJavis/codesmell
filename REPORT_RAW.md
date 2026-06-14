# DeepSmells-CodeBERT — Raw-source code smell detection
### Strategy 3 · fine-tune CodeBERT on raw `.code` (C# + Java, 4 smells)

**Idea:** Drop the paper's "token-index → 1D-CNN" entirely. Feed **raw source code** to a
pretrained code language model (**CodeBERT**) and fine-tune a classifier per (language, smell).

---

## 1. Why this is different from the paper

The paper *explicitly* avoids learned embeddings — *"our work encodes the whole source code by
tokenization indexing"* — feeding bare integer indices to a CNN. CodeBERT instead brings:
- a real **sub-word code tokenizer** (handles unseen identifiers),
- **pretrained** transformer weights (knows code structure from 6 languages),
- contextual **embeddings** learned on millions of functions.

So this tests: *does a modern pretrained code model on raw source beat the hand-rolled
token-index pipeline?*

---

## 2. Data & setup

Source: Sharma `training_data_{cs,java}` raw `.code` fragments (same origin as the lab data).
Per (lang, smell): all positives (cap 8,000) + **5× negatives** → `data_raw/<lang>_<smell>.jsonl`.

| | samples | positives |
|---|---:|---:|
| cs ComplexMethod | 48,000 | 8,000 |
| cs ComplexConditional | 36,534 | 6,089 |
| cs FeatureEnvy | 13,122 | 2,187 |
| cs MultifacetedAbstraction | 2,316 | 386 |
| java ComplexMethod | 48,000 | 8,000 |
| java ComplexConditional | 48,000 | 8,000 |
| java FeatureEnvy | 20,574 | 3,429 |
| java MultifacetedAbstraction | 5,082 | 847 |

Model: `microsoft/codebert-base` + classification head, `max_len=512`, 3 epochs, AdamW 2e-5,
fp16, class-weighted cross-entropy, stratified 70/30, threshold tuned on validation.
Compute: ~2.5 h on RTX 5060 Ti (8 models).

---

## 3. Results — all 8 (lang × smell)

| lang | smell | Precision | Recall | F1 | MCC |
|---|---|---:|---:|---:|---:|
| cs | ComplexConditional | 0.8711 | 0.9102 | **0.8903** | 0.8680 |
| cs | ComplexMethod | 0.8068 | 0.9450 | **0.8705** | 0.8460 |
| cs | FeatureEnvy | 0.7643 | 0.8552 | **0.8072** | 0.7678 |
| cs | MultifacetedAbstraction | 0.7075 | 0.8966 | **0.7909** | 0.7508 |
| java | ComplexConditional | 0.9401 | 0.9679 | **0.9538** | 0.9446 |
| java | ComplexMethod | 0.9259 | 0.9679 | **0.9464** | 0.9358 |
| java | FeatureEnvy | 0.6838 | 0.7629 | **0.7212** | 0.6632 |
| java | MultifacetedAbstraction | 0.6688 | 0.8268 | **0.7394** | 0.6864 |

All combos reach F1 0.72–0.95, MCC 0.66–0.94.

---

## 4. ⚠ Comparison with the paper — read the caveat first

Paper DeepSmells (Table 2, **C#**, on the **true** class imbalance):

| Smell | Paper F1 | Paper MCC | Our cs F1 | Our cs MCC |
|---|---:|---:|---:|---:|
| ComplexMethod | 0.7542 | 0.7341 | 0.8705 | 0.8460 |
| ComplexConditional | 0.5892 | 0.5684 | 0.8903 | 0.8680 |
| FeatureEnvy | 0.2940 | 0.2686 | 0.8072 | 0.7678 |
| MultifacetedAbstraction | 0.2793 | 0.2752 | 0.7909 | 0.7508 |

**Do NOT read this as a clean 3× win on FE/MA.** The numbers are on **different evaluation
distributions**:
- Paper evaluates at the **true imbalance** (FE ≈ 0.7% positive, MA ≈ 0.13%) — brutally hard, which
  is why their FE/MA F1 is ~0.28.
- Ours evaluates at the **sampled 1:5 ratio (~17% positive)** — far easier, especially for the rare
  smells. Much of our FE/MA lead is the easier eval, **not** purely model quality.
- For CM/CC the eval gap is smaller (paper ~8%/4% vs our 17%), so the lead there is more meaningful
  but still partly distribution-driven.

**Honest takeaways:**
1. CodeBERT on raw source is a **strong, working detector** for all 4 smells in both languages.
2. The cross-paper table is **indicative, not head-to-head** — the eval imbalance differs.
3. The clean, defensible win is **architectural**: raw-source + pretrained embeddings vs the
   paper's token-index CNN, and it removes the paper's stated limitation (no embeddings).

A true head-to-head would re-run CodeBERT at the paper's full imbalance (inference over millions of
negatives) — out of scope here.

---

## 5. Cross-language observation (C# vs Java)

- **Java > C#** on the complexity smells: java CC 0.954 / CM 0.946 vs cs 0.890 / 0.871 — java had
  more positives (8k capped from 20k–45k) and CodeBERT was pretrained on more Java.
- **C# > Java** on FeatureEnvy (0.807 vs 0.721) and MultifacetedAbstraction (0.791 vs 0.739) at the
  combo level — but counts differ, so treat as observation, not conclusion.
- Pattern matches the paper's note that rarer smells (FE, MA) are the hardest.

---

## 6. Limitations

- **Eval imbalance** (1:5) inflates absolute scores vs paper, most for FE/MA (see §4).
- **`max_len=512`**: long methods are truncated — CodeBERT cannot see the tail of large
  ComplexMethods. A sliding-window or long-context model could help.
- Threshold tuned on the validation set (same set used for model selection) — mildly optimistic.
- No separate held-out test; checkpoints not saved (8×500 MB) — results in `results_codebert.json`.

---

## 7. Deliverables

| File | What |
|---|---|
| `build_dataset.py` | raw `.7z` → sampled `data_raw/*.jsonl` (8 combos) |
| `codebert_smells.py` | fine-tune CodeBERT per combo, P/R/F1/MCC + threshold |
| `results_codebert.json` | all 8 results |
| `REPORT_RAW.md` | this report |
| spec | `docs/superpowers/specs/2026-06-14-codebert-raw-source-design.md` |

**Conclusion:** Fine-tuning CodeBERT on raw source code detects all four smells across C# and Java
with F1 0.72–0.95. It cleanly addresses the paper's design choice (no learned embeddings) and far
outperforms our token-index baseline. The headline gap vs the paper's published numbers is real but
partly an artifact of a less-imbalanced evaluation set — stated plainly rather than oversold.
