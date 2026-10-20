"""接続診断: 各データサイトがこの環境から見られるかを data/diag.json に記録する。"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from .http import DEFAULT_UA

def _targets() -> dict:
    from . import config as C
    h = C.load()["hall"]
    return {
        "anaslo_list": h.get("anaslo_list_url", ""),
        "anaslo_top": "https://ana-slo.com/",
        "minrepo_tag": h.get("minrepo_tag_url", ""),
    }


def _summ(status, headers, text):
    title = re.search(r"<title[^>]*>(.*?)</title>", text or "", re.S | re.I)
    body = re.sub(r"<script.*?</script>|<style.*?</style>", " ", text or "", flags=re.S | re.I)
    body = re.sub(r"<[^>]+>", " ", body)
    body = re.sub(r"\s+", " ", body).strip()
    h = {k.lower(): v for k, v in (headers or {}).items()}
    return {"status": status, "title": (title.group(1).strip()[:120] if title else ""),
            "server": h.get("server", ""), "cf_mitigated": h.get("cf-mitigated", ""),
            "len": len(text or ""), "snippet": body[:400]}


def run(root: Path) -> dict:
    out = {"when": datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M"), "requests": {}, "browser": {}}
    s = requests.Session()
    s.headers.update({"User-Agent": DEFAULT_UA, "Accept-Language": "ja,en;q=0.8"})
    targets = _targets()
    for k, u in targets.items():
        try:
            r = s.get(u, timeout=40)
            out["requests"][k] = _summ(r.status_code, r.headers, r.text)
        except Exception as e:
            out["requests"][k] = {"error": str(e)[:200]}
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch(headless=True)
            pg = b.new_context(user_agent=DEFAULT_UA, locale="ja-JP").new_page()
            for k in ("anaslo_list",):
                try:
                    resp = pg.goto(targets[k], wait_until="domcontentloaded", timeout=45000)
                    pg.wait_for_timeout(6000)
                    out["browser"][k] = _summ(resp.status if resp else 0, resp.headers if resp else {}, pg.content())
                except Exception as e:
                    out["browser"][k] = {"error": str(e)[:200]}
            b.close()
    except Exception as e:
        out["browser"]["error"] = str(e)[:200]
    try:
        out["egress_ip"] = requests.get("https://api.ipify.org", timeout=15).text
    except Exception:
        pass
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "diag.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
