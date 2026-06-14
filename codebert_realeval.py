"""CodeBERT fine-tune with TRUE-IMBALANCE evaluation (fair-er vs paper Table 2).
Train: sampled (all train-positives + 5x negatives) for tractability.
Val/Test: held-out, built at the TRUE positive ratio (disjoint negatives), so P/R/F1/MCC
reflect the real class imbalance like the paper. Threshold tuned on val, reported on test."""
import py7zr, json, random, shutil, glob, os, time
import numpy as np, torch, torch.nn as nn
from sklearn import metrics
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL='microsoft/codebert-base'; MAX_LEN=512; EPOCHS=3; BATCH=16; LR=2e-5; SEED=0
TRAIN_POS_CAP=8000; NEG_RATIO_TRAIN=5
EVAL_POS_CAP=3000           # held-out positives for eval (split val/test 50/50)
VAL_NEG_CAP=15000           # smaller val (threshold tuning every epoch)
TEST_NEG_CAP=40000          # test at true ratio, evaluated ONCE at best epoch (RAM-bounded)
LANGS=['cs','java']; SMELLS=['ComplexMethod','ComplexConditional','FeatureEnvy','MultifacetedAbstraction']
TMP='/tmp/re_extract'
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

print('device',dev, torch.cuda.get_device_name(0) if dev.type=='cuda' else '', flush=True)
tok=AutoTokenizer.from_pretrained(MODEL)

def plan(lang, smell):
    """Decide target file names for train/val/test with true-ratio eval."""
    rng=random.Random(SEED)
    with py7zr.SevenZipFile(f'raw/{lang}/{smell}.7z','r') as z:
        names=[n for n in z.getnames() if n.endswith('.code')]
    pos=[n for n in names if '/Positive/' in n]; neg=[n for n in names if '/Negative/' in n]
    rng.shuffle(pos); rng.shuffle(neg)
    P,N=len(pos),len(neg); ratio=N/P  # negatives per positive (true)
    n_eval_pos=min(int(0.3*P), EVAL_POS_CAP)
    eval_pos=pos[:n_eval_pos]; train_pos=pos[n_eval_pos:][:TRAIN_POS_CAP]
    val_pos=eval_pos[:len(eval_pos)//2]; test_pos=eval_pos[len(eval_pos)//2:]
    n_train_neg=min(NEG_RATIO_TRAIN*len(train_pos), N)
    n_val_neg=min(int(round(len(val_pos)*ratio)), VAL_NEG_CAP)
    n_test_neg=min(int(round(len(test_pos)*ratio)), TEST_NEG_CAP)
    # disjoint slices from shuffled neg
    need=n_train_neg+n_val_neg+n_test_neg
    if need>N:  # shrink eval negs proportionally
        scale=(N-n_train_neg)/(n_val_neg+n_test_neg)
        n_val_neg=int(n_val_neg*scale); n_test_neg=int(n_test_neg*scale)
    a=n_train_neg; b=a+n_val_neg; c=b+n_test_neg
    train_neg=neg[:a]; val_neg=neg[a:b]; test_neg=neg[b:c]
    label={};
    for L in (train_pos,val_pos,test_pos):
        for n in L: label[n]=1
    for L in (train_neg,val_neg,test_neg):
        for n in L: label[n]=0
    groups={'train':set(train_pos)|set(train_neg),'val':set(val_pos)|set(val_neg),'test':set(test_pos)|set(test_neg)}
    return label, groups, dict(P=P,N=N,ratio=round(ratio,1),
        train=len(train_pos)+len(train_neg), train_pos=len(train_pos),
        val_pos=len(val_pos), val_neg=len(val_neg),
        test_pos=len(test_pos), test_neg=len(test_neg),
        test_pospct=round(100*len(test_pos)/max(len(test_pos)+len(test_neg),1),3))

def extract_read(lang, smell, label):
    shutil.rmtree(TMP, ignore_errors=True)
    with py7zr.SevenZipFile(f'raw/{lang}/{smell}.7z','r') as z:
        z.extract(path=TMP, targets=list(label.keys()))
    data={}
    for fp in glob.glob(f'{TMP}/**/*.code', recursive=True):
        rel=os.path.relpath(fp,TMP).replace(os.sep,'/')
        if rel in label:
            t=open(fp,errors='ignore').read().strip()
            if t: data[rel]=t
    shutil.rmtree(TMP, ignore_errors=True)
    return data

def encode(codes):
    e=tok(codes,truncation=True,max_length=MAX_LEN,padding='max_length',return_tensors='pt')
    return e['input_ids'],e['attention_mask']

def batches(ids,mask,y,bs,shuffle):
    idx=np.arange(len(ids))
    if shuffle: np.random.shuffle(idx)
    for i in range(0,len(idx),bs):
        j=idx[i:i+bs]
        yield ids[j],mask[j],(None if y is None else torch.tensor(y[j],dtype=torch.long))

@torch.no_grad()
def predict(model,ids,mask,bs=64):
    model.eval(); out=[]
    for bi,bm,_ in batches(ids,mask,None,bs,False):
        with torch.amp.autocast('cuda',enabled=dev.type=='cuda'):
            lg=model(input_ids=bi.to(dev),attention_mask=bm.to(dev)).logits
        out.append(torch.softmax(lg.float(),-1)[:,1].cpu().numpy())
    return np.concatenate(out)

def scores(p,y,t):
    pr=p>=t
    return (metrics.precision_score(y,pr,zero_division=0),metrics.recall_score(y,pr,zero_division=0),
            metrics.f1_score(y,pr,zero_division=0),metrics.matthews_corrcoef(y,pr))

def run(lang,smell):
    set_seed(SEED)
    label,groups,info=plan(lang,smell)
    print(f'[{lang}_{smell}] plan',info,flush=True)
    data=extract_read(lang,smell,label)
    def build(g):
        items=[(c,label[n]) for n,c in data.items() if n in groups[g]]
        codes=[c for c,_ in items]; y=np.array([l for _,l in items])
        return codes,y
    trC,trY=build('train'); vC,vY=build('val'); teC,teY=build('test')
    ids_tr,m_tr=encode(trC); ids_v,m_v=encode(vC); ids_te,m_te=encode(teC)
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,num_labels=2).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=LR)
    scaler=torch.amp.GradScaler('cuda',enabled=dev.type=='cuda')
    posw=(len(trY)-trY.sum())/max(trY.sum(),1)
    lossf=nn.CrossEntropyLoss(weight=torch.tensor([1.0,float(posw)],device=dev))
    # threshold sweep tuned on val (true ratio) each epoch; keep best-val model, test ONCE at end
    bestv={'valf1':-1}; best_state=None; best_thr=0.5; best_ep=0
    for ep in range(EPOCHS):
        model.train(); t0=time.time()
        for bi,bm,by in batches(ids_tr,m_tr,trY,BATCH,True):
            opt.zero_grad()
            with torch.amp.autocast('cuda',enabled=dev.type=='cuda'):
                loss=lossf(model(input_ids=bi.to(dev),attention_mask=bm.to(dev)).logits,by.to(dev))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        pv=predict(model,ids_v,m_v)
        bt,bf=0.5,-1
        for t in np.linspace(0.05,0.95,19):
            f=metrics.f1_score(vY,pv>=t,zero_division=0)
            if f>bf: bt,bf=t,f
        print(f'[{lang}_{smell}] ep{ep+1} valF1={bf:.4f} thr={bt:.2f} ({time.time()-t0:.0f}s)',flush=True)
        if bf>bestv['valf1']:
            bestv={'valf1':bf}; best_thr=float(bt); best_ep=ep+1
            best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
    # final TEST eval at true ratio, on best-val model
    model.load_state_dict(best_state); model.to(dev)
    pte=predict(model,ids_te,m_te)
    P,R,F1,MCC=scores(pte,teY,best_thr)
    print(f'[{lang}_{smell}] FINAL TEST F1={F1:.4f} P={P:.4f} R={R:.4f} MCC={MCC:.4f} thr={best_thr:.2f} ep={best_ep}',flush=True)
    del model; torch.cuda.empty_cache()
    return {'lang':lang,'smell':smell,**info,'precision':P,'recall':R,'f1':F1,'mcc':MCC,'threshold':best_thr,'epoch':best_ep}

if __name__=='__main__':
    RES='results_codebert_realeval.json'
    results=json.load(open(RES)) if os.path.exists(RES) else []
    done={(r['lang'],r['smell']) for r in results}
    print('resume: already done', sorted(done), flush=True)
    for lang in LANGS:
        for smell in SMELLS:
            if (lang,smell) in done:
                print('skip (done)', lang, smell, flush=True); continue
            try:
                r=run(lang,smell); results.append(r); print('  ->',r,flush=True)
            except Exception as e:
                print('FAIL',lang,smell,repr(e)[:160],flush=True)
            json.dump(results,open(RES,'w'),indent=2)
    print('\n=== TRUE-IMBALANCE RESULTS ===')
    for r in results:
        print(f"{r['lang']:5} {r['smell']:24} testpos%={r['test_pospct']:6} P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} MCC={r['mcc']:.4f}")
    print('DONE')
