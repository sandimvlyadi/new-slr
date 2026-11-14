"""Fetch metadata + abstracts for IEEE records not in Scopus via OpenAlex (DOI), fallback Crossref.
Applies the same DL + G1-G4 keyword blocks as the Scopus/PubMed queries (IEEE query had only task + radiograph blocks).
Output: search/ieee_records.csv (all 216, with match flag)."""
import csv, json, re, time, urllib.parse, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DL = re.compile(r'deep learning|neural network|convolutional|\bcnn|transformer|machine learning', re.I)
G = re.compile(r'self-supervised|self supervised|contrastive|masked autoencoder|masked image model|simclr|moco|byol|\bdino|pretext|foundation model|pre-train|pretrain|transfer learning|imbalanc|long-tail|long tail|focal loss|oversampl|undersampl|over-sampl|under-sampl|class weight|weighted loss|cost-sensitive|smote|ordinal regression|ordinal loss|ordinal classification|mixup|data augmentation|calibrat|uncertaint|conformal|monte carlo dropout|deep ensemble|evidential|bayesian|selective prediction|out-of-distribution|abstention|reject option|confidence estimat|label noise|noisy label|label correction|label smoothing|teacher-student|teacher student|mean teacher|knowledge distillation|pseudo-label|pseudo label|semi-supervised|semi supervised|co-teaching|inter-rater|interobserver|inter-observer|label uncertainty|soft label|weak supervision|weakly supervised', re.I)
UA = {'User-Agent': 'slr-methods-script/0.1'}
def get(u):
    return json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60))
def inv(ii):
    if not ii: return ''
    pos = sorted((p, w) for w, ps in ii.items() for p in ps)
    return ' '.join(w for _, w in pos)
dois = [l.strip() for l in open(ROOT / 'search/ieee_nonscopus_dois.txt') if l.strip().startswith('10.')]
recs = {}
for i in range(0, len(dois), 40):
    chunk = dois[i:i+40]
    u = 'https://api.openalex.org/works?' + urllib.parse.urlencode({'filter': 'doi:' + '|'.join(chunk), 'per-page': 50,
        'select': 'doi,title,publication_year,type,primary_location,abstract_inverted_index,keywords'})
    for w in get(u)['results']:
        d = (w.get('doi') or '').replace('https://doi.org/', '').lower()
        src = ((w.get('primary_location') or {}).get('source') or {}).get('display_name', '')
        recs[d] = {'doi': d, 'title': w.get('title') or '', 'year': w.get('publication_year'), 'type': w.get('type'),
                   'source': src, 'abstract': inv(w.get('abstract_inverted_index')),
                   'keywords': '; '.join(k.get('display_name', '') for k in (w.get('keywords') or []))}
    time.sleep(0.3)
for d in dois:
    if d in recs and recs[d]['abstract']: continue
    try:
        m = get('https://api.crossref.org/works/' + urllib.parse.quote(d))['message']
        r = recs.setdefault(d, {'doi': d, 'title': (m.get('title') or [''])[0], 'year': (m.get('issued', {}).get('date-parts') or [[None]])[0][0],
                                'type': m.get('type'), 'source': (m.get('container-title') or [''])[0], 'abstract': '', 'keywords': ''})
        r['abstract'] = re.sub(r'<[^>]+>', ' ', m.get('abstract', '') or '').strip()
        if not r['title']: r['title'] = (m.get('title') or [''])[0]
    except Exception as e:
        recs.setdefault(d, {'doi': d, 'title': '', 'year': '', 'type': '', 'source': '', 'abstract': '', 'keywords': ''})
    time.sleep(0.2)
rows = []
for d in dois:
    r = recs[d]; t = f"{r['title']} {r['abstract']} {r['keywords']}"
    r['has_abstract'] = bool(r['abstract']); r['match_dl_g'] = bool(DL.search(t) and G.search(t))
    rows.append(r)
with open(ROOT / 'search/ieee_records.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('records', len(rows), 'with abstract', sum(r['has_abstract'] for r in rows), 'match', sum(r['match_dl_g'] for r in rows))
print('no abstract + title-only match candidates', sum(1 for r in rows if not r['has_abstract']))
