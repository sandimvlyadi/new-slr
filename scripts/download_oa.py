"""Download OA full texts for full-text candidates (is_oa=True) into papers/<ID>.pdf.
Sources tried in order: OpenAlex best_oa pdf_url, Europe PMC PDF render (PMCID), all OpenAlex OA locations.
Files are untrusted data: only saved and checked for the %PDF header, never executed."""
import csv, json, time, urllib.parse, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
UA = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) slr-methods-script/0.1', 'Accept': 'application/pdf,*/*'}
def fetch(u):
    with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60) as r:
        return r.read()
def oa_urls(doi):
    try:
        w = json.load(urllib.request.urlopen(urllib.request.Request('https://api.openalex.org/works/doi:' + urllib.parse.quote(doi), headers=UA), timeout=60))
        return [l.get('pdf_url') for l in w.get('locations', []) if l.get('is_oa') and l.get('pdf_url')]
    except Exception:
        return []
rows = [r for r in csv.DictReader(open(ROOT / 'screening/02_fulltext_candidates.csv')) if r['is_oa'] == 'True']
log = []
for r in rows:
    out = ROOT / 'papers' / f"{r['id']}.pdf"
    if out.exists() and out.read_bytes()[:4] == b'%PDF':
        log.append((r['id'], 'exists', '')); continue
    cands = [r['pdf_url']] if r['pdf_url'] else []
    if r['pmcid']:
        cands.append(f"https://europepmc.org/api/fulltextRepo?pprId=&type=FILE&fileName={r['pmcid']}.pdf")
        cands.append(f"https://www.ncbi.nlm.nih.gov/pmc/articles/{r['pmcid']}/pdf/")
    if r['doi']:
        cands += [u for u in oa_urls(r['doi']) if u not in cands]
    ok = ''
    for u in cands:
        try:
            b = fetch(u)
            if b[:4] == b'%PDF':
                out.write_bytes(b); ok = u; break
        except Exception:
            pass
        time.sleep(0.5)
    log.append((r['id'], 'ok' if ok else 'FAILED', ok or ' | '.join(cands)[:200]))
with open(ROOT / 'papers/download_log.tsv', 'w') as f:
    for x in log: f.write('\t'.join(x) + '\n')
print('ok', sum(1 for x in log if x[1] in ('ok', 'exists')), 'failed', sum(1 for x in log if x[1] == 'FAILED'))
for x in log:
    if x[1] == 'FAILED': print(x[0], x[2])
