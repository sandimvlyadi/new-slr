"""Print abstract snippets around G-term matches for given record ids."""
import csv, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
P = re.compile(r'self-supervised|contrastive|masked|foundation|pre-?train\w*|transfer learning|imbalanc\w*|focal|over-?sampl\w*|under-?sampl\w*|class weight\w*|weighted|smote|ordinal|mixup|augment\w*|calibrat\w*|uncertaint\w*|conformal|dropout|ensemble|evidential|bayesian|label|teacher|distillation|semi-?supervised|weakly|inter-?observer|inter-?rater|self-learning|freez\w*|fine-?tun\w*|grad\w* |kellgren|mri|x-?ray|radiograph', re.I)
rows = {r['id']: r for r in csv.DictReader(open(ROOT / 'screening/00_merged_records.csv'))}
W = int(sys.argv[2]) if len(sys.argv) > 2 else 90
for i in sys.argv[1].split(','):
    r = rows[i]; a = r['abstract']; spans = []
    for m in P.finditer(a):
        s, e = max(0, m.start() - W), min(len(a), m.end() + W)
        if spans and s <= spans[-1][1]: spans[-1][1] = e
        else: spans.append([s, e])
    print(f"## {i} {r['title'][:90]}\n" + ' … '.join(a[s:e] for s, e in spans[:4])[:700] + '\n')
