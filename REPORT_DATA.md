# Data Processing — Đời sống của một sample
### DeepSmells / ComplexMethod (1d) · truy vết bằng số thật từ dataset

Report này theo chân **một dòng dữ liệu thật** đi qua từng bước của pipeline, kèm con số thật đo
trực tiếp trên `data/tokenizer_cs/ComplexMethod/1d`. Mục tiêu: hiểu *chính xác* dữ liệu biến đổi
thế nào trước khi vào model.

---

## 0. Token là gì? (đọc được bằng mắt)

Mỗi sample = **1 dòng** trong file `.tok.cld`, là chuỗi số nguyên cách nhau bởi dấu cách. Số = token.
Cách mã hoá (suy ra từ data thật):

- **Id nhỏ 33–126 = mã ASCII của ký tự đơn** (dấu câu / toán tử):
  `40='('  41=')'  123='{'  125='}'  59=';'  61='='  46='.'  60='<'  62='>'  33='!'`
- **Id lớn (≥128, vd 392, 2001, 2002…) = chỉ số từ điển** của identifier / keyword / literal.
  Mỗi định danh khác nhau → một số khác nhau.

> ⚠ Điểm mấu chốt: số token là **categorical** (nhãn), KHÔNG phải độ lớn. `2001` không "lớn hơn"
> `40` về mặt ngữ nghĩa. Đây chính là lý do baseline (nhét số thô vào Conv1d) yếu, và vì sao
> DeepSmells+ thêm `Embedding` — xem `REPORT_IMPROVED.md`.

---

## 1. Bước 1 — Đọc 1 dòng → mảng số

**Sample Positive thật** (dòng đầu `Positive/tokenized1.tok.cld`), **714 token**:
```
123 392 40 33 40 2001 407 424 41 41 123 392 40 2001 46 2002 60 1502 41 450 59 490 2003 61 418 ...
  {   .   (   !   (  id   .   .   )   )   {   .   (  id   .  id   <  id   )  id   ;  id  id   =  id ...
```
**Sample Negative thật** (33 token — method ngắn, sạch):
```
123 2001 61 473 59 2002 61 418 2003 60 2004 62 40 41 59 ... 125
  {  id   =  id   ;  id   =  id  id   <  id   >   (   )   ;  ...   }
```
Code: `np.fromstring(line, dtype=int32, sep=' ')`. Dòng text → mảng `int`. Độ dài mảng = số token.

---

## 2. Bước 2 — Đo độ dài mọi sample → phân phối

Quét toàn bộ file, ghi độ dài từng dòng:

| | #samples | mean len | std len | max len |
|---|---:|---:|---:|---:|
| Positive | 26,164 | 390.2 | 681.7 | 57,012 |
| Negative | 466,503 | 91.5 | 654.0 | 221,890 |

Nhận xét: Positive (method có smell) **dài hơn nhiều** (mean 390 vs 91) — hợp lý, "Complex Method"
thường dài. Nhưng có outlier cực đoan (max 57,012 token!) kéo lệch.

---

## 3. Bước 3 — Cắt outlier → `max_input_length`

Không thể pad mọi sample tới 57,012 (phí RAM khủng khiếp). Cắt theo **mean + 1·std**:

```
cutoff_pos = 390.2 + 681.7 = 1071.9   → max token giữ lại (Positive) = 1071
cutoff_neg =  91.5 + 654.0 =  745.4   → max token giữ lại (Negative) =  745
max_input_length = max(1071, 745)     = 1071     ← dùng chung cho cả hai
```
- Positive: bỏ **1,204** dòng dài > cutoff (giữ 24,960).
- `max_input_length = 1071` → mọi sample sau này dài đúng 1071.

---

## 4. Bước 4 — Lọc theo `max_input_length` rồi zero-pad

Đọc lại data, **chỉ giữ** sample có `0 < len ≤ 1071`, rồi đệm số 0 cho đủ 1071:

```
sample Positive: 714 token thật  →  [123, 392, 40, ..., 125, 125,  0, 0, 0, ..., 0]
                                     └──── 714 số thật ────┘ └──── 357 số 0 ────┘
                                                  tổng = 1071
```
Kiểm chứng dòng thật: sau pad, `nonzero = 714`, đuôi = `[125, 125, 0, 0, 0]`. Pad **bằng 0** vì 0
không trùng token thật nào (token nhỏ nhất trong data = 33) → 0 trở thành "padding token" an toàn.

Số sample còn lại sau lọc (`len ≤ 1071`):
- Positive: **24,960** / 26,164
- Negative: **464,861** / 466,503

---

## 5. Bước 5 — Gom thành mảng NumPy

Danh sách các mảng (mỗi cái dài 1071) → một mảng lớn, rồi thêm trục kênh:

```
list[N] of (1071,)  →  reshape  →  (N, 1071, 1)
```
`1` cuối = số kênh (channel) cho Conv1d. Label: Positive = 1.0, Negative = 0.0.

---

## 6. Bước 6 — Cân bằng train + giới hạn (số thật)

Tỉ lệ train/eval = 0.7. Với 24,960 pos và 464,861 neg:

```
train_pos = 0.7 × 24,960  = 17,472        eval_pos = 24,960  − 17,472 =   7,488
train_neg = 0.7 × 464,861 = 325,402       eval_neg = 464,861 − 325,402 = 139,459
```
**Cân bằng + cap train** (lab dùng `MAX_TRAINING_SAMPLES = 5000`):
```
train_pos = train_neg = min(5000, 17472, 325402) = 5,000
```
→ train pool ban đầu = 5,000 pos + 5,000 neg = **10,000** (cân bằng 50/50, để model không "lười"
đoán toàn Negative).
Eval **không cân bằng** (giữ thực tế): 7,488 pos + 139,459 neg = **146,947** (chỉ ~5% pos).
*(max_eval=None → trần 150,000 cho CM; 139,459 < 150,000 nên không cắt thêm.)*

---

## 7. Bước 7 — Gộp lại + chia stratified (số thật)

Lab **gộp** train+eval rồi **chia lại** stratified 70/30 (giữ tỉ lệ nhãn ở cả hai phần):

```
all = 10,000 + 146,947 = 156,947 sample   (pos = 5,000 + 7,488 = 12,488  → 7.96%)
                          │ stratified 70/30
        ┌─────────────────┴──────────────────┐
   train = 109,862                       valid = 47,085
   (pos 7.96%)                           (pos 7.96%)   ← imbalance THẬT, dùng để báo cáo
```
Đây đúng là con số `train_data (109862,1071,1)`, `valid_data (47085,1071,1)`, `pos 7.96%` thấy khi chạy.

> Lưu ý: vì gộp+chia lại, nhiều negative của "eval" lọt vào train → train phình lên 109,862 (chủ yếu
> negative). Tốt cho việc học imbalance thật. Caveat về việc này so với protocol paper: xem
> `REPORT_IMPROVED.md` §1.1 và §9.

---

## 8. Bước 8 — Dataset → Tensor → Batch

`CodeSmellDataset.__getitem__` lấy 1 sample, đổi sang tensor:

**Baseline** (số thô, float):
```
(1071, 1) float  → reshape → (1, 1071) float     # 1 kênh cho Conv1d
DataLoader gộp 128 sample →  batch  (128, 1, 1071),  label (128, 1)
```
**DeepSmells+** (cho Embedding, int):
```
(1071,) long   →  batch  (128, 1071) long
Embedding(8463, 32, pad=0):  (128, 1071)  →  (128, 1071, 32)
permute:                     (128, 1071, 32) → (128, 32, 1071)   # 32 kênh vào Conv1d
```

---

## 9. Tóm tắt — 1 sample đổi shape thế nào

| Giai đoạn | Ví dụ Positive | Shape |
|---|---|---|
| Dòng text | `"123 392 40 ..."` | string |
| Parse | `[123,392,40,...]` 714 số | (714,) |
| Lọc ≤1071 | giữ (714 ≤ 1071) | (714,) |
| Zero-pad | + 357 số 0 | (1071,) |
| Thêm kênh | | (1071, 1) |
| Trong batch (baseline) | | (128, 1, 1071) |
| Sau Embedding (improved) | | (128, 32, 1071) |

---

## 10. Vì sao mỗi bước cần thiết (chốt cho thuyết trình)

- **Parse số**: model chỉ ăn số, không ăn text.
- **Đo độ dài + cắt outlier**: vài method khổng lồ (57k token) sẽ ép pad toàn bộ → nổ RAM; cắt ở
  mean+std giữ ~95% sample mà chặn chi phí.
- **Pad về cùng độ dài**: một batch là một tensor → mọi dòng phải cùng shape.
- **Cân bằng train**: chống bias "đoán toàn Negative" trên data lệch.
- **Giữ valid lệch thật**: đo đúng năng lực thực tế (không tự thưởng điểm bằng eval cân bằng giả).
- **Stratified split**: train & valid cùng tỉ lệ nhãn → đánh giá ổn định.

> Dữ liệu này (`tokenizer_cs`) đến từ link Kaggle trong gói đề; cùng họ benchmark với Sharma et al.
> mà paper DeepSmells dùng, nhưng số lượng khác (xem `REPORT_IMPROVED.md` §1.1).
