"""台データオンライン（daidata.goraggio.com）のページ構成を調べる（調査用）。"""
import json, re, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

HALL = "100305"
BASE = f"https://daidata.goraggio.com/{HALL}"
out = Path("data/debug/daidata"); out.mkdir(parents=True, exist_ok=True)
urls = [BASE, f"{BASE}/all_list?ps=S"]
log = []
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    ctx = b.new_context(locale="ja-JP", user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1",
                        viewport={"width": 390, "height": 844})
    pg = ctx.new_page()
    def grab(u, name):
        try:
            r = pg.goto(u, wait_until="domcontentloaded", timeout=45000)
            pg.wait_for_timeout(2500)
            # 規約同意のボタンがあれば押す
            for t in ("同意する", "同意して", "OK", "閉じる"):
                try:
                    el = pg.get_by_text(t, exact=False).first
                    if el.is_visible(timeout=500):
                        el.click(); pg.wait_for_timeout(2000)
                except Exception:
                    pass
            html = pg.content()
            (out / f"{name}.html").write_text(html, encoding="utf-8")
            links = re.findall(r'href="([^"]+)"', html)
            log.append({"url": u, "final": pg.url, "status": r.status if r else None, "len": len(html),
                        "title": pg.title(), "links": [l for l in links if "goraggio" in l or l.startswith("/")][:200]})
            return html
        except Exception as e:
            log.append({"url": u, "error": str(e)[:300]})
            return ""
    for i, u in enumerate(urls):
        grab(u, f"p{i}")
    # ハナハナの機種リンクを探して1つ開く → 台リンクを1つ開く
    html = (out / "p1.html").read_text(encoding="utf-8") if (out / "p1.html").exists() else ""
    hana = [h for h in re.findall(r'href="([^"]+)"[^>]*>(?:(?!</a>).)*?ハナハナ', html, re.S)]
    if hana:
        u = hana[0] if hana[0].startswith("http") else "https://daidata.goraggio.com" + hana[0]
        h2 = grab(u.replace("&amp;", "&"), "p2_model")
        units = re.findall(r'href="([^"]*detail[^"]*)"', h2)
        if units:
            u3 = units[0] if units[0].startswith("http") else "https://daidata.goraggio.com" + units[0]
            grab(u3.replace("&amp;", "&"), "p3_unit")
    b.close()
(out / "log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(log, ensure_ascii=False)[:3000])
