"""DeepSmells+ — improved CM detector: Embedding + Focal Loss + AdamW + early stop + threshold tuning.
Reuses the lab data pipeline (cell 13 of codesmell-slab.ipynb) verbatim."""
import json, random, time, sys
import numpy as np, torch, torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader

SMELL, DIM = 'ComplexMethod', '1d'
DATA_ROOT = Path('data/tokenizer_cs')
SEED = 0
VOCAB = 8463          # token ids 33..8462 -> need 8463 (pad idx 0)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- reuse the exact lab data loader (defines get_all_data, InputData, set_seed, ...) ---
_nb = json.load(open('codesmell-slab.ipynb'))
exec(''.join(_nb['cells'][13]['source']))

# ---------------- data ----------------
def build_data(max_eval_samples, max_training_samples=5000, train_bs=256, valid_bs=512):
    # Hold data identical to the baseline reproduce so the validation set keeps the
    # true ~8% imbalance -> improvement is attributable to model/loss/threshold, not data.
    set_seed(SEED)
    d = get_all_data(DATA_ROOT, SMELL, DIM, max_training_samples=max_training_samples,
                     max_eval_samples=max_eval_samples, seed=SEED)
    class DS(Dataset):
        def __init__(s, X, y): s.X, s.y = X, y
        def __len__(s): return len(s.X)
        def __getitem__(s, i):
            return (torch.tensor(s.X[i].reshape(-1), dtype=torch.long),
                    torch.tensor([s.y[i]], dtype=torch.float32))
    tl = DataLoader(DS(d.train_data, d.train_labels), batch_size=train_bs, shuffle=True, num_workers=2)
    vl = DataLoader(DS(d.eval_data, d.eval_labels), batch_size=valid_bs, shuffle=False, num_workers=2)
    return d, tl, vl

# ---------------- model ----------------
def conv_out(L, k, s=1): return (L - (k - 1) - 1) // s + 1
def calc_lstm(L, k):
    for _ in range(2):
        L = conv_out(L, k); L = conv_out(L, 2, 2)
    return L

class DeepSmellsPlus(nn.Module):
    def __init__(self, vocab, embed_dim, seq_len, kernel=5, hidden=100, c1=16, c2=32, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, c1, kernel), nn.BatchNorm1d(c1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel), nn.BatchNorm1d(c2), nn.ReLU(), nn.MaxPool1d(2))
        Lp = calc_lstm(seq_len, kernel)
        self.lstm = nn.LSTM(input_size=Lp, hidden_size=hidden, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(64, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x):
        e = self.embed(x).permute(0, 2, 1)   # B x embed x L
        o = self.conv(e)                      # B x c2 x L'
        _, (h, _) = self.lstm(o)
        return self.fc(h[-1])                 # B x 1

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__(); self.a, self.g = alpha, gamma
    def forward(self, logits, targets):
        ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        at = self.a * targets + (1 - self.a) * (1 - targets)
        return (at * (1 - pt) ** self.g * ce).mean()

# ---------------- eval helpers ----------------
@torch.no_grad()
def predict(model, loader):
    model.eval(); P, Y = [], []
    for x, y in loader:
        P.append(torch.sigmoid(model(x.to(dev))).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P).ravel(), np.concatenate(Y).ravel()

def best_threshold(probs, y):
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.05, 0.95, 19):
        f1 = metrics.f1_score(y, probs >= t, zero_division=0)
        if f1 > best_f1: best_t, best_f1 = t, f1
    return best_t

def score(probs, y, t):
    pred = probs >= t
    return (metrics.precision_score(y, pred, zero_division=0),
            metrics.recall_score(y, pred, zero_division=0),
            metrics.f1_score(y, pred, zero_division=0),
            metrics.matthews_corrcoef(y, pred))

# ---------------- train one config ----------------
def train_cfg(tl, vl, seq_len, embed_dim, gamma, alpha, kernel=5, max_epochs=40, patience=6, tag=''):
    set_seed(SEED)
    model = DeepSmellsPlus(VOCAB, embed_dim, seq_len, kernel).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = FocalLoss(alpha, gamma)
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda')
    best = {'f1': -1}; best_state = None; bad = 0
    for ep in range(max_epochs):
        model.train(); t0 = time.time()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev); opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=dev.type == 'cuda'):
                loss = lossf(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        probs, yv = predict(model, vl)
        t = best_threshold(probs, yv)
        P, R, F1, MCC = score(probs, yv, t)
        print(f'[{tag}] ep{ep+1:02d} F1={F1:.4f} P={P:.4f} R={R:.4f} MCC={MCC:.4f} thr={t:.2f} ({time.time()-t0:.0f}s)', flush=True)
        if F1 > best['f1']:
            best = {'precision': P, 'recall': R, 'f1': F1, 'mcc': MCC, 'threshold': float(t), 'epoch': ep + 1}
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= patience: print(f'[{tag}] early stop @ep{ep+1}', flush=True); break
    return best, best_state

if __name__ == '__main__':
    print('device', dev, torch.cuda.get_device_name(0) if dev.type == 'cuda' else '')
    # Same data as baseline: cap train positives at 5000, full eval -> realistic ~8% imbalance.
    d, tl, vl = build_data(max_eval_samples=None, max_training_samples=5000)
    SEQ = d.max_input_length
    print('SEQ', SEQ, 'train', d.train_data.shape, 'valid', d.eval_data.shape,
          'pos%', round(float(d.eval_labels.mean()) * 100, 2), flush=True)
    # small search (<=8): embed_dim x gamma x alpha (kernel fixed 5)
    search = [(32, 2.0, 0.75), (64, 2.0, 0.75), (64, 1.0, 0.75),
              (64, 2.0, 0.5), (32, 2.0, 0.5), (64, 3.0, 0.75)]
    results = []
    best_states = {}
    for emb, g, a in search:
        tag = f'e{emb}_g{g}_a{a}'
        best, state = train_cfg(tl, vl, SEQ, emb, g, a, max_epochs=40, patience=6, tag=tag)
        results.append({'embed_dim': emb, 'gamma': g, 'alpha': a, **best})
        best_states[tag] = state
        print('  ->', tag, best, flush=True)
    results.sort(key=lambda r: -r['f1'])
    json.dump(results, open('results_improved.json', 'w'), indent=2)
    print('\n=== RESULTS (sorted F1, full ~8% eval) ===')
    for r in results: print(r)
    win = results[0]
    wtag = f"e{win['embed_dim']}_g{win['gamma']}_a{win['alpha']}"
    torch.save({'state_dict': best_states[wtag], 'config': win}, 'deepsmells_plus_best.pth')
    print('\n=== WINNER (final) ===', win, flush=True)
