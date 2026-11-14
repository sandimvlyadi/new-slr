"""OA status for full-text candidates via OpenAlex (by DOI, else title search)."""
import csv, json, time, urllib.parse, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
UA = {'User-Agent': 'slr-methods-script/0.1'}
get = lambda u: json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60))
rows = [r for r in csv.DictReader(open(ROOT / 'screening/01_title_abstract_screening.csv')) if r['decision'] == 'include_fulltext']
out = []
for r in rows:
    w = None
    try:
        if r['doi']:
            w = get('https://api.openalex.org/works/doi:' + urllib.parse.quote(r['doi']))
        else:
            res = get('https://api.openalex.org/works?' + urllib.parse.urlencode({'search': r['title'][:200], 'per-page': 1}))['results']
            w = res[0] if res else None
    except Exception:
        w = None
    oa = (w or {}).get('open_access') or {}
    ids = (w or {}).get('ids') or {}
    best = (w or {}).get('best_oa_location') or {}
    src = (((w or {}).get('primary_location') or {}).get('source') or {})
    out.append({'id': r['id'], 'groups': r['groups_or_reason'], 'year': r['year'], 'doi': r['doi'], 'title': r['title'][:120],
                'is_oa': oa.get('is_oa'), 'oa_status': oa.get('oa_status'), 'pmcid': (ids.get('pmcid') or '').split('/')[-1],
                'pdf_url': best.get('pdf_url') or '', 'landing': best.get('landing_page_url') or '',
                'venue': src.get('display_name', ''), 'issn_l': src.get('issn_l') or '', 'cited_by': (w or {}).get('cited_by_count')})
    time.sleep(0.15)
with open(ROOT / 'screening/02_fulltext_candidates.csv', 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=list(out[0])); wr.writeheader(); wr.writerows(out)
import collections
print(len(out), collections.Counter(str(o['is_oa']) for o in out), 'pmc', sum(1 for o in out if o['pmcid']))
