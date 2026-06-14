"""Build compact JSONL datasets from Sharma raw .code archives.
Per (lang, smell): all positives (cap 8000) + 5x negatives, targeted-extracted from .7z."""
import py7zr, json, random, shutil, glob, os, time
from pathlib import Path

LANGS = ['cs', 'java']
SMELLS = ['ComplexMethod', 'ComplexConditional', 'FeatureEnvy', 'MultifacetedAbstraction']
MAX_POS = 8000
NEG_RATIO = 5
SEED = 0
OUT = Path('data_raw'); OUT.mkdir(exist_ok=True)
TMP = Path('/tmp/cs_extract')

def build(lang, smell):
    arc = f'raw/{lang}/{smell}.7z'
    out = OUT / f'{lang}_{smell}.jsonl'
    if out.exists():
        print('skip (exists)', out); return
    rng = random.Random(SEED)
    with py7zr.SevenZipFile(arc, 'r') as z:
        names = [n for n in z.getnames() if n.endswith('.code')]
    pos = [n for n in names if '/Positive/' in n]
    neg = [n for n in names if '/Negative/' in n]
    pos_keep = pos if len(pos) <= MAX_POS else rng.sample(pos, MAX_POS)
    n_neg = min(NEG_RATIO * len(pos_keep), len(neg))
    neg_keep = rng.sample(neg, n_neg)
    targets = pos_keep + neg_keep
    labels = {n: 1 for n in pos_keep}; labels.update({n: 0 for n in neg_keep})

    shutil.rmtree(TMP, ignore_errors=True)
    t = time.time()
    with py7zr.SevenZipFile(arc, 'r') as z:
        z.extract(path=str(TMP), targets=targets)
    n_written = 0
    with out.open('w') as f:
        for fp in glob.glob(str(TMP / '**' / '*.code'), recursive=True):
            rel = os.path.relpath(fp, TMP).replace(os.sep, '/')
            lab = labels.get(rel)
            if lab is None:
                continue
            code = open(fp, errors='ignore').read().strip()
            if not code:
                continue
            f.write(json.dumps({'code': code, 'label': lab}) + '\n')
            n_written += 1
    shutil.rmtree(TMP, ignore_errors=True)
    print(f'{lang:5} {smell:24} pos={len(pos_keep):5} neg={len(neg_keep):6} written={n_written:6} ({time.time()-t:.0f}s)', flush=True)

if __name__ == '__main__':
    for lang in LANGS:
        for smell in SMELLS:
            build(lang, smell)
    print('DATASET BUILD DONE')
    for f in sorted(glob.glob('data_raw/*.jsonl')):
        n = sum(1 for _ in open(f))
        print(f, n, 'samples')
