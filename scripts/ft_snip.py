"""Targeted full-text snippets per paper for extraction (dataset, split, ablation, baseline, external, code, compute, metrics)."""
import csv, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CAT = {
 'DATA': r'\b(OAI|MOST|Osteoarthritis Initiative|Kaggle|Mendeley|dataset (?:consist|contain|compris|includ)|\d[\d,]{2,} (?:images|radiographs|knees|patients|joints|subjects))',
 'SPLIT': r'(patient[- ]level|per[- ]patient|subject[- ]wise|patient[- ]wise|split|five[- ]fold|5-fold|\bfold\b|hold[- ]out|external (?:test|validation|cohort|dataset))',
 'ABL': r'(ablation|w/o|without the|removing the|component analysis|contribution of each)',
 'BASE': r'(baseline|compared (?:with|to)|outperform|improv\w+ (?:of|by)|increase\w* (?:of|by|from)|from \d+\.\d+ ?%? to \d)',
 'METRIC': r'(quadratic weighted kappa|\bQWK\b|\bkappa\b|balanced accuracy|macro[- ]F1|macro[- ]averag|\bMAE\b|\bECE\b|coverage|AUROC|\bAUC\b)',
 'CODE': r'(github|gitlab|code (?:is|will be) (?:publicly )?available|available at|zenodo|huggingface)',
 'COMP': r'(GPU|NVIDIA|RTX|Tesla|V100|A100|GeForce|training time|epochs?\b|batch size|learning rate)',
 'SEED': r'(random seeds?|\bseeds?\b|repeated (?:\d+|five|three|ten) times|mean ± ?|± ?\d|standard deviation|confidence interval|95% CI|p ?[<=] ?0\.\d)',
}
def snip(t, pat, n, w=110):
    out, last = [], -10**9
    for m in re.finditer(pat, t, re.I):
        if m.start() - last < w * 2: continue
        s, e = max(0, m.start() - w), min(len(t), m.end() + w)
        out.append(re.sub(r'\s+', ' ', t[s:e]).strip()); last = m.start()
        if len(out) >= n: break
    return out
ids = sys.argv[1].split(','); N = {'DATA':4,'SPLIT':4,'ABL':4,'BASE':5,'METRIC':3,'CODE':2,'COMP':3,'SEED':3}
meta = {r['id']: r for r in csv.DictReader(open(ROOT / 'screening/02_fulltext_candidates.csv'))}
for i in ids:
    t = (ROOT / f'fulltext/{i}.txt').read_text(errors='ignore')
    t = re.split(r'\n\s*(?:REFERENCES|References|Bibliography)\s*\n', t)[0]
    print(f"######## {i} | {meta[i]['venue'][:50]} | {meta[i]['title'][:100]} | words={len(t.split())}")
    for c, p in CAT.items():
        s = snip(t, p, N[c])
        if s: print(f"[{c}] " + ' || '.join(s)[:900])
    print()
