"""One round of backward + forward citation searching (OpenAlex) from included studies with an ablation (Q1_abl=Y).
Keeps candidates 2021-2026 not already in merged records; pre-filters by task + DL keywords in title/abstract (same blocks as the database search).
Output: search/citation_candidates.csv"""
import csv, json, re, time, urllib.parse, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
UA = {'User-Agent': 'slr-methods-script/0.1'}
get = lambda u: json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=90))
TASK = re.compile(r'knee osteoarthritis|hand osteoarthritis|kellgren|\bKL grad|rheumatoid arthritis|sharp score|van der heijde|bone erosion|joint space narrowing|oarsi|mtss|total sharp', re.I)
XR = re.compile(r'radiograph|x-?ray|plain film', re.I)
DL = re.compile(r'deep learning|neural network|convolutional|\bcnn|transformer|machine learning', re.I)
def inv(ii):
    if not ii: return ''
    return ' '.join(w for _, w in sorted((p, w) for w, ps in ii.items() for p in ps))
norm = lambda t: re.sub(r'[^a-z0-9]', '', (t or '').lower())[:80]
merged = list(csv.DictReader(open(ROOT / 'screening/00_merged_records.csv')))
known_doi = {r['doi'].lower() for r in merged if r['doi']}; known_t = {norm(r['title']) for r in merged}
ext = list(csv.DictReader(open(ROOT / 'extraction/01_fulltext_extraction.tsv'), delimiter='\t'))
seeds = [r['id'] for r in ext if r['Q1_abl'] == 'Y']
cand = {r['id']: r for r in csv.DictReader(open(ROOT / 'screening/02_fulltext_candidates.csv'))}
SEL = 'id,doi,title,publication_year,type,primary_location,abstract_inverted_index,referenced_works'
found = {}
def add(w, how, seed):
    d = (w.get('doi') or '').replace('https://doi.org/', '').lower(); t = w.get('title') or ''
    y = w.get('publication_year') or 0
    if not (2021 <= y <= 2026) or (d and d in known_doi) or norm(t) in known_t: return
    a = inv(w.get('abstract_inverted_index')); txt = f'{t} {a}'
    if not (TASK.search(txt) and XR.search(txt) and DL.search(txt)): return
    k = d or norm(t)
    r = found.setdefault(k, {'doi': d, 'title': t, 'year': y, 'type': w.get('type'),
         'source': (((w.get('primary_location') or {}).get('source') or {}).get('display_name') or ''), 'abstract': a, 'via': set()})
    r['via'].add(f'{how}:{seed}')
nb = nf = 0
for s in seeds:
    doi = cand[s]['doi']
    try:
        w = get('https://api.openalex.org/works/doi:' + urllib.parse.quote(doi)) if doi else \
            get('https://api.openalex.org/works?' + urllib.parse.urlencode({'search': cand[s]['title'][:150], 'per-page': 1}))['results'][0]
    except Exception:
        continue
    refs = [x.split('/')[-1] for x in w.get('referenced_works', [])]
    for i in range(0, len(refs), 50):
        try:
            res = get('https://api.openalex.org/works?' + urllib.parse.urlencode({'filter': 'openalex:' + '|'.join(refs[i:i+50]), 'per-page': 50, 'select': SEL}))['results']
        except Exception:
            res = []
        nb += len(res); [add(x, 'back', s) for x in res]; time.sleep(0.2)
    wid = w['id'].split('/')[-1]; cursor = '*'
    while cursor:
        try:
            res = get('https://api.openalex.org/works?' + urllib.parse.urlencode({'filter': f'cites:{wid}', 'per-page': 200, 'select': SEL, 'cursor': cursor}))
        except Exception:
            break
        nf += len(res['results']); [add(x, 'fwd', s) for x in res['results']]
        cursor = res['meta'].get('next_cursor'); time.sleep(0.2)
        if not res['results']: break
rows = [dict(r, via=';'.join(sorted(r['via']))) for r in found.values()]
for i, r in enumerate(rows, 1): r['id'] = f'C{i:03d}'
with open(ROOT / 'search/citation_candidates.csv', 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=['id', 'doi', 'title', 'year', 'type', 'source', 'via', 'abstract']); wr.writeheader(); wr.writerows(rows)
print('seeds', len(seeds), 'backward records', nb, 'forward records', nf, 'new candidates after prefilter', len(rows))
