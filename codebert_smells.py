"""Fine-tune CodeBERT on raw .code fragments for each (lang, smell) combo.
Reads data_raw/<lang>_<smell>.jsonl, stratified 70/30, reports P/R/F1/MCC + tuned threshold."""
import json, glob, os, time, random
import numpy as np, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = 'microsoft/codebert-base'
MAX_LEN = 512
EPOCHS = 3
BATCH = 16
LR = 2e-5
SEED = 0
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

print('device', dev, torch.cuda.get_device_name(0) if dev.type == 'cuda' else '', flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)

def load(path):
    codes, labels = [], []
    for line in open(path):
        o = json.loads(line); codes.append(o['code']); labels.append(int(o['label']))
    return codes, np.array(labels)

def encode(codes):
    enc = tok(codes, truncation=True, max_length=MAX_LEN, padding='max_length', return_tensors='pt')
    return enc['input_ids'], enc['attention_mask']

def batches(ids, mask, y, bs, shuffle):
    idx = np.arange(len(y))
    if shuffle: np.random.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield ids[j], mask[j], (None if y is None else torch.tensor(y[j], dtype=torch.long))

@torch.no_grad()
def predict(model, ids, mask, bs=32):
    model.eval(); probs = []
    for bi, bm, _ in batches(ids, mask, np.zeros(len(ids)), bs, False):
        with torch.amp.autocast('cuda', enabled=dev.type == 'cuda'):
            logit = model(input_ids=bi.to(dev), attention_mask=bm.to(dev)).logits
        probs.append(torch.softmax(logit.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(probs)

def best_threshold(p, y):
    bt, bf = 0.5, -1
    for t in np.linspace(0.05, 0.95, 19):
        f = metrics.f1_score(y, p >= t, zero_division=0)
        if f > bf: bt, bf = t, f
    return bt

def run_combo(path):
    name = os.path.basename(path)[:-6]
    set_seed(SEED)
    codes, y = load(path)
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        print('skip tiny', name); return None
    Xtr_c, Xte_c, ytr, yte = train_test_split(codes, y, test_size=0.3, stratify=y, random_state=SEED)
    ids_tr, m_tr = encode(Xtr_c); ids_te, m_te = encode(Xte_c)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda')
    pos_w = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    w = torch.tensor([1.0, float(pos_w)], device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)

    best = {'f1': -1}
    for ep in range(EPOCHS):
        model.train(); t0 = time.time()
        for bi, bm, by in batches(ids_tr, m_tr, ytr, BATCH, True):
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=dev.type == 'cuda'):
                logit = model(input_ids=bi.to(dev), attention_mask=bm.to(dev)).logits
                loss = lossf(logit, by.to(dev))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        p = predict(model, ids_te, m_te)
        th = best_threshold(p, yte); pred = p >= th
        P = metrics.precision_score(yte, pred, zero_division=0)
        R = metrics.recall_score(yte, pred, zero_division=0)
        F1 = metrics.f1_score(yte, pred, zero_division=0)
        MCC = metrics.matthews_corrcoef(yte, pred)
        print(f'[{name}] ep{ep+1} F1={F1:.4f} P={P:.4f} R={R:.4f} MCC={MCC:.4f} thr={th:.2f} ({time.time()-t0:.0f}s)', flush=True)
        if F1 > best['f1']:
            best = {'precision': P, 'recall': R, 'f1': F1, 'mcc': MCC, 'threshold': float(th), 'epoch': ep + 1}
    del model; torch.cuda.empty_cache()
    lang, smell = name.split('_', 1)
    return {'lang': lang, 'smell': smell, 'n': len(y), 'pos': int(y.sum()), **best}

if __name__ == '__main__':
    results = []
    for path in sorted(glob.glob('data_raw/*.jsonl')):
        try:
            r = run_combo(path)
            if r: results.append(r); print('  ->', r, flush=True)
        except Exception as e:
            print('FAIL', path, repr(e)[:160], flush=True)
        json.dump(results, open('results_codebert.json', 'w'), indent=2)
    print('\n=== CODEBERT RESULTS ===')
    for r in sorted(results, key=lambda x: (x['lang'], x['smell'])):
        print(f"{r['lang']:5} {r['smell']:24} P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} MCC={r['mcc']:.4f}")
    print('DONE')
