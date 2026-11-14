"""q.py ID[,ID] 'regex' [width] [max] : whitespace-normalised regex snippets from fulltext/ID.txt"""
import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
w = int(sys.argv[3]) if len(sys.argv) > 3 else 150; mx = int(sys.argv[4]) if len(sys.argv) > 4 else 4
for i in sys.argv[1].split(','):
    t = re.sub(r'\s+', ' ', (ROOT / f'fulltext/{i}.txt').read_text(errors='ignore'))
    for k, m in enumerate(re.finditer(sys.argv[2], t, re.I)):
        if k >= mx: break
        print(f"{i}: …{t[max(0,m.start()-w):m.end()+w]}…")
