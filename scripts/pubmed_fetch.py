"""Fetch PubMed records for SLR-M: query in search/pubmed_query.txt -> search/pubmed_records.csv."""
import csv, json, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
E = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
q = (ROOT / 'search/pubmed_query.txt').read_text().strip()
r = json.load(urllib.request.urlopen(E + 'esearch.fcgi?' + urllib.parse.urlencode(
    {'db': 'pubmed', 'term': q, 'retmax': 1000, 'retmode': 'json'})))
ids = r['esearchresult']['idlist']
print('count', r['esearchresult']['count'], 'ids', len(ids))
rows = []
for i in range(0, len(ids), 100):
    x = urllib.request.urlopen(E + 'efetch.fcgi?' + urllib.parse.urlencode(
        {'db': 'pubmed', 'id': ','.join(ids[i:i+100]), 'retmode': 'xml'})).read()
    for a in ET.fromstring(x).findall('.//PubmedArticle'):
        g = lambda p: ''.join(a.find(p).itertext()).strip() if a.find(p) is not None else ''
        doi = next((e.text for e in a.findall('.//ArticleId') if e.get('IdType') == 'doi'), '')
        abst = ' '.join(''.join(t.itertext()) for t in a.findall('.//Abstract/AbstractText'))
        au = '; '.join(f"{n.findtext('LastName','')} {n.findtext('Initials','')}".strip() for n in a.findall('.//AuthorList/Author'))
        pt = '; '.join(p.text for p in a.findall('.//PublicationType'))
        kw = '; '.join(k.text or '' for k in a.findall('.//Keyword'))
        rows.append({'pmid': g('.//PMID'), 'doi': (doi or '').lower(), 'title': g('.//ArticleTitle'),
                     'year': g('.//PubDate/Year') or g('.//PubDate/MedlineDate')[:4], 'journal': g('.//Journal/Title'),
                     'issn': g('.//ISSN'), 'authors': au, 'pubtype': pt, 'keywords': kw, 'abstract': abst})
    time.sleep(0.4)
with open(ROOT / 'search/pubmed_records.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('written', len(rows))
