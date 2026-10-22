"""東三河のハナハナ設置店の調査: みんレポから各店のハナハナの台ごとの数字（直近90日・最大45日分）を集める。

出力: data/survey/halls.json（店の一覧）, data/survey/rows.csv（店・日・機種・台番・G数・差枚・ボーナス合計）
途中で止まっても、次に動かすと取っていない店・日から続ける。
"""
import csv, json, os, re, sys, time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import quote, unquote
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hana.http import Fetcher, RateLimited
from hana.sources import minrepo as MR

CITIES = ["豊川市", "豊橋市", "蒲郡市", "新城市", "田原市"]
_cfg = json.loads(Path("tools/survey_config.json").read_text(encoding="utf-8")) if Path("tools/survey_config.json").exists() else {}
DAYS_BACK, MAX_DAYS = int(_cfg.get("days_back", 90)), int(_cfg.get("max_days", 45))
KEY = _cfg.get("keyword", "ハナハナ")                 # 機種名にこの文字を含む機種を集める
ONLY = set(_cfg.get("halls", []))                     # 空なら全部の店
OUT = Path(_cfg.get("out", "data/survey")); OUT.mkdir(parents=True, exist_ok=True)
ROWS = OUT / "rows.csv"; HALLS = OUT / "halls.json"; DONE = OUT / "done.json"
deadline = time.time() + 300 * 60
today = date.today()
f = Fetcher(interval_sec=2, jitter_sec=1, engine="playwright")

# 1) 店の一覧
halls = json.loads(HALLS.read_text(encoding="utf-8")) if HALLS.exists() else {}
if not halls and Path("data/survey/halls.json").exists():
    halls = json.loads(Path("data/survey/halls.json").read_text(encoding="utf-8"))
if ONLY:
    halls = {k: v for k, v in halls.items() if v["name"] in ONLY}
if not halls:
    for c in CITIES:
        base = f"https://min-repo.com/category/{quote('愛知県')}/{quote(c)}/"
        for p in range(1, 6):
            url = base if p == 1 else f"{base}page/{p}/"
            try:
                h = f.get(url)
            except Exception as e:
                print("一覧の取得失敗", url, e); break
            found = 0
            for u, t in re.findall(r'<a[^>]+href="(https://min-repo\.com/tag/[^"]+)"[^>]*>(.*?)</a>', h, re.S):
                name = re.sub(r"<[^>]+>", "", t).strip()
                if name and u not in halls:
                    halls[u] = {"name": name, "city": c}; found += 1
            if f"/page/{p + 1}/" not in h:
                break
    HALLS.write_text(json.dumps(halls, ensure_ascii=False, indent=1), encoding="utf-8")
print("店の数:", len(halls))

done = json.loads(DONE.read_text(encoding="utf-8")) if DONE.exists() else {}
new_file = not ROWS.exists()
fh = open(ROWS, "a", newline="", encoding="utf-8"); w = csv.writer(fh)
if new_file:
    w.writerow(["hall", "city", "date", "model", "unit", "games", "diff", "total"])

for tag, info in halls.items():
    if time.time() > deadline:
        break
    st = done.get(tag, {})
    if st.get("finished"):
        continue
    name = info["name"]
    try:
        h = f.get(tag)
        _, links, last = MR.parse_tag_page(h, tag, today)
        if last >= 2 and len(links) < MAX_DAYS:
            h2 = f.get(f"{tag.rstrip('/')}/page/2/", referer=tag)
            links += MR.parse_tag_page(h2, tag, today)[1]
    except Exception as e:
        print("タグの取得失敗", name, e); continue
    since = (today - timedelta(days=DAYS_BACK)).isoformat()
    links = sorted({(d, u) for d, u in links if d >= since}, reverse=True)[:MAX_DAYS]
    got = set(st.get("dates", []))
    todo = [(d, u) for d, u in links if d not in got]
    print(f"{name}: 対象{len(links)}日 未取得{len(todo)}日", flush=True)
    has_hana, checked = st.get("has_hana"), 0
    for i in range(0, len(todo), 3):
        if time.time() > deadline:
            break
        chunk = todo[i:i + 3]
        try:
            pages = f.get_many([u for _, u in chunk], referer=tag, parallel=3)
        except RateLimited:
            print("  アクセス制限。少し休みます"); time.sleep(60); continue
        except Exception as e:
            print("  日別ページ失敗", e); continue
        kurls, meta = [], []
        for (d, u), html in zip(chunk, pages):
            d = MR.post_date(html, d)
            models, variety = MR.parse_day_models(html, KEY)
            checked += 1
            for v in variety:
                w.writerow([name, info["city"], d, v["model"], v["unit"], v["games"], v["diff"], ""])
            for m in models:
                kurls.append(f"{u.rstrip('/')}/?kishu={quote(m)}"); meta.append((d, m))
            if models or variety:
                has_hana = True
            got.add(d)
        if kurls:
            try:
                kp = f.get_many(kurls, referer=chunk[0][1], parallel=3)
            except Exception as e:
                print("  機種ページ失敗", e); kp = []
            for (d, m), html in zip(meta, kp):
                for r in MR.parse_kishu_page(html):
                    w.writerow([name, info["city"], d, m, r["unit"], r["games"], r["diff"], r["total"]])
        fh.flush()
        if not has_hana and checked >= 3:      # 3日見てハナハナが無ければ、この店は対象外
            break
    done[tag] = {"name": name, "dates": sorted(got), "has_hana": bool(has_hana),
                 "finished": (not has_hana and checked >= 3) or len([1 for d, _ in links if d not in got]) == 0}
    DONE.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → ハナハナ{'あり' if has_hana else 'なし'} 取得済み{len(got)}日", flush=True)
fh.close(); f.close()
print("おわり", sum(1 for v in done.values() if v.get("finished")), "/", len(halls), "店")
