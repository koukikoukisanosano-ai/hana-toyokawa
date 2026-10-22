"""みんレポの市区ページの構成を調べる（東三河のハナハナ店調査の準備）。"""
import json, re, sys
from pathlib import Path
from urllib.parse import quote
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hana.http import Fetcher

CITIES = ["豊川市", "豊橋市", "蒲郡市", "新城市", "田原市"]
out = {}
f = Fetcher(interval_sec=2, jitter_sec=1, engine="playwright")
for c in CITIES:
    url = f"https://min-repo.com/category/{quote('愛知県')}/{quote(c)}/"
    try:
        h = f.get(url)
    except Exception as e:
        out[c] = {"url": url, "error": str(e)[:200]}
        continue
    links = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', h, re.S)
    links = [(u, re.sub(r"<[^>]+>", "", t).strip()[:60]) for u, t in links]
    out[c] = {"url": url, "len": len(h),
              "tag_links": [x for x in links if "/tag/" in x[0]][:200],
              "post_links": [x for x in links if re.search(r"min-repo\.com/\d+/?$", x[0])][:60],
              "page_links": [x[0] for x in links if "/page/" in x[0]][:20],
              "head": re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", h))[:1500]}
f.close()
Path("data/survey").mkdir(parents=True, exist_ok=True)
Path("data/survey/probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: {"len": v.get("len"), "tags": len(v.get("tag_links", [])), "posts": len(v.get("post_links", []))} for k, v in out.items()}, ensure_ascii=False))
