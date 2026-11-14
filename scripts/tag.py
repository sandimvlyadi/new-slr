"""Tag merged records with matched G1-G4 terms, task and modality hints (aid for screening, not a decision)."""
import csv, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
GR = {
 'G1': r'self-supervised|self supervised|contrastive|masked autoencoder|masked image model|simclr|moco|byol|\bdino|pretext|foundation model|pre-?train\w*|transfer learning',
 'G2': r'imbalanc\w*|long-?tail\w*|focal loss|over-?sampl\w*|under-?sampl\w*|class weight\w*|weighted loss|cost-sensitive|smote|ordinal (?:regression|loss|classification)|mixup|data augmentation',
 'G3': r'calibrat\w*|uncertaint\w*|conformal|monte carlo dropout|deep ensemble\w*|evidential|bayesian|selective prediction|out-of-distribution|abstention|reject option|confidence estimat\w*',
 'G4': r'label noise|noisy label\w*|label correction|label smoothing|teacher-?student|mean teacher|knowledge distillation|pseudo-? ?label\w*|semi-? ?supervised|co-teaching|inter-?rater|inter-?observer|interobserver|label uncertainty|soft label\w*|weak(?:ly)? supervis\w*'}
TASK = r'kellgren|\bkl\b|oarsi|sharp|svdh|mtss|erosion|joint space narrowing|\bjsn\b|severity|grad(?:e|ing)'
MOD = r'mri|magnetic resonance|ultrasound|\bct\b|computed tomography|radiograph\w*|x-?ray\w*'
rows = list(csv.DictReader(open(ROOT / 'screening/00_merged_records.csv')))
for r in rows:
    t = f"{r['title']} {r['abstract']}".lower()
    tags = []
    for g, p in GR.items():
        m = sorted(set(x if isinstance(x, str) else x[0] for x in re.findall(p, t)))
        if m: tags.append(g + ':' + ','.join(m)[:60])
    r['_tags'] = ' '.join(tags)
    r['_task'] = ','.join(sorted(set(re.findall(TASK, t))))[:40]
    r['_mod'] = ','.join(sorted(set(re.sub(r'\w*$', '', x) if x.startswith(('radiograph','x')) else x for x in re.findall(MOD, t))))[:30]
lo, hi = int(sys.argv[1]), int(sys.argv[2])
for r in rows[lo:hi]:
    print(f"{r['id']}|{r['year']}|{r['doctype'][:4]}|{r['title'][:110]}|{r['_tags']}|T:{r['_task']}|M:{r['_mod']}")
