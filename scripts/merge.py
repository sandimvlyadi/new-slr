"""Merge Scopus + PubMed + IEEE(non-Scopus, DL+G filtered or no abstract) -> screening/00_merged_records.csv."""
import csv, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
norm = lambda t: re.sub(r'[^a-z0-9]', '', (t or '').lower())[:80]
out, seen_doi, seen_t = [], {}, {}
def add(src, doi, title, year, source, dtype, abstract, kw, issn=''):
    doi = (doi or '').lower().strip(); k = norm(title)
    hit = seen_doi.get(doi) if doi else None
    if hit is None: hit = seen_t.get(k)
    if hit is not None:
        r = out[hit]; r['sources'] += ';' + src
        if not r['abstract'] and abstract: r['abstract'] = abstract
        if not r['doi'] and doi: r['doi'] = doi; seen_doi[doi] = hit
        return
    out.append({'id': '', 'sources': src, 'doi': doi, 'title': title, 'year': year, 'source': source, 'doctype': dtype,
                'issn': issn, 'keywords': kw, 'abstract': abstract})
    i = len(out) - 1
    if doi: seen_doi[doi] = i
    seen_t[k] = i
n = {}
for r in csv.DictReader(open(ROOT / 'search/scopus_export_2026-11-10.csv', encoding='utf-8-sig')):
    add('scopus', r['DOI'], r['Title'], r['Year'], r['Source title'], r['Document Type'], r['Abstract'],
        '; '.join(x for x in (r['Author Keywords'], r['Index Keywords']) if x), r['ISSN'])
n['scopus'] = 290
for r in csv.DictReader(open(ROOT / 'search/pubmed_records.csv')):
    add('pubmed', r['doi'], r['title'], r['year'], r['journal'], r['pubtype'], r['abstract'], r['keywords'], r['issn'])
for r in csv.DictReader(open(ROOT / 'search/ieee_records.csv')):
    if r['match_dl_g'] == 'True' or r['has_abstract'] == 'False':
        add('ieee', r['doi'], r['title'], r['year'], r['source'], r['type'], r['abstract'], r['keywords'])
for i, r in enumerate(out, 1): r['id'] = f'M{i:03d}'
(ROOT / 'screening').mkdir(exist_ok=True)
with open(ROOT / 'screening/00_merged_records.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
import collections
print('unique', len(out)); print(collections.Counter(r['sources'] for r in out))
