"""みんレポ（min-repo.com）からの取得。

・オーギヤWOのページ: https://min-repo.com/tag/wo/ （2021/10〜。2026/7/30 以降は更新が止まっている）
・日別ページ → 機種ページ(?kishu=機種名) に台番ごとの 差枚/G数/合成確率
・台番ページ(?num=台番) に BB/RB 回数

使い方は2通り:
  1) 補完（precise）: アナスロに無い新しい日があれば、台番ページまで開いて BB/RB を正確に取る
  2) 過去分（estimate）: アナスロより古い日を、機種ページだけ（1日1〜2ページ）で取り、
     BB/RB は 差枚・G数・合成確率 から推定する（推定値として区別して保存）
"""
from __future__ import annotations

import logging
import re
import time
from datetime import date as Date, timedelta
from urllib.parse import quote, urljoin

import numpy as np
from bs4 import BeautifulSoup

from ..http import Fetcher, RateLimited
from .anaslo import to_int

log = logging.getLogger("hana.minrepo")

_MD = re.compile(r"(?:(\d{4})/)?(\d{1,2})/(\d{1,2})\((.)\)")
_PUB = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
_WD = "月火水木金土日"


def infer_year(month: int, day: int, weekday_char: str, today: Date) -> int | None:
    for y in (today.year, today.year - 1, today.year - 2):
        try:
            d = Date(y, month, day)
        except ValueError:
            continue
        if d <= today and _WD[d.weekday()] == weekday_char:
            return y
    return None


def parse_tag_page(html: str, base_url: str, today: Date) -> tuple[list[dict], list[tuple[str, str]], int]:
    """タグページ → (日別サマリー, [(date, 日別URL)], 最終ページ番号)"""
    soup = BeautifulSoup(html, "lxml")
    days, links = [], []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue
        m = _MD.search(tds[0].get_text())
        a = tds[0].find("a", href=True)
        if not m:
            continue
        y = int(m.group(1)) if m.group(1) else infer_year(int(m.group(2)), int(m.group(3)), m.group(4), today)
        if not y:
            continue
        d = f"{y:04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        days.append({"date": d, "total_diff": to_int(tds[1].get_text()), "avg_diff": to_int(tds[2].get_text()),
                     "avg_games": to_int(tds[3].get_text())})
        if a and re.search(r"min-repo\.com/\d+/?$", urljoin(base_url, a["href"])):
            links.append((d, urljoin(base_url, a["href"])))
    last_page = 1
    for a in soup.find_all("a", href=True):
        m = re.search(r"/page/(\d+)/?$", a["href"])
        if m:
            last_page = max(last_page, int(m.group(1)))
    return days, links, last_page


def _num(s):
    """みんレポの数値。「-1.234」のようにピリオドが桁区切りのことがあるので直してから読む。"""
    t = _nk(str(s or "")).strip().replace("‐", "-").replace("−", "-")
    if re.fullmatch(r"[+\-]?\d{1,3}(\.\d{3})+", t.replace("枚", "").replace(",", "")):
        t = t.replace(".", "")
    return to_int(t)


def _nk(s: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKC", s or "")


def debug_dump(html: str, url: str) -> dict:
    """機種が見つからなかった日別ページの様子（原因調査用）"""
    soup = BeautifulSoup(html, "lxml")
    kishu = [(a.get_text(strip=True), a["href"]) for a in soup.find_all("a", href=True) if "kishu" in a["href"]]
    heads = [[th.get_text(strip=True) for th in t.find_all("th")][:8] for t in soup.find_all("table")][:15]
    return {"url": url, "len": len(html), "title": soup.title.get_text() if soup.title else "",
            "kishu_links": kishu[:120], "table_heads": heads,
            "text": soup.get_text(" ", strip=True)[:3000], "raw": html[:1500]}


def parse_day_models(html: str, keyword: str) -> tuple[list[str], list[dict]]:
    """日別ページ → (対象機種名リスト, バラエティ行[{unit, model, diff, games}])"""
    keyword = _nk(keyword)
    soup = BeautifulSoup(html, "lxml")
    models = []
    for a in soup.find_all("a", href=True):
        if "kishu=" in a["href"] and "kishu=all" not in a["href"] and keyword in _nk(a.get_text()):
            name = a.get_text(strip=True)
            if name not in models:
                models.append(name)
    variety = []
    for table in soup.find_all("table"):
        head = [th.get_text(strip=True) for th in table.find_all("th")]
        if head[:3] == ["機種", "台番", "差枚"]:
            for tr in table.find_all("tr")[1:]:
                tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                if len(tds) >= 4 and keyword in _nk(tds[0]) and tds[0] not in models:
                    variety.append({"model": tds[0], "unit": to_int(tds[1]), "diff": to_int(tds[2]),
                                    "games": to_int(tds[3])})
    return models, variety


def parse_kishu_page(html: str) -> list[dict]:
    """機種ページ → [{unit, diff, games, total}]（total は合成確率から逆算したボーナス合計）"""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for li in soup.select("ul.slump_list > li"):
        a = li.find("a")
        table = li.find("table")
        if not a or not table:
            continue
        unit = to_int(a.get_text())
        tds = [td.get_text(strip=True) for td in table.find_all("td")]
        if unit is None or len(tds) < 3:
            continue
        diff, games = _num(tds[0]), _num(tds[1])
        m = re.search(r"1/([\d.]+)", tds[2])
        total = None
        if m and games:
            denom = float(m.group(1))
            total = int(round(games / denom)) if denom > 0 else 0
        out.append({"unit": unit, "diff": diff, "games": games, "total": total})
    return out


def parse_unit_page(html: str) -> dict | None:
    """台番ページ → {model, diff, games, bb, rb}"""
    soup = BeautifulSoup(html, "lxml")
    res = {}
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        head = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
        vals = [c.get_text(strip=True) for c in rows[1].find_all(["th", "td"])]
        if head[:2] == ["機種", "差枚"] and len(vals) >= 3:
            res.update(model=vals[0], diff=_num(vals[1]), games=_num(vals[2]), diff_raw=vals[1])
        elif head[:2] == ["BB", "RB"] and len(vals) >= 2:
            res.update(bb=to_int(vals[0]), rb=to_int(vals[1]))
    return res if "bb" in res else None


def parse_unit_history(html: str, today: Date) -> list[dict]:
    """台番ページの「日付・差枚・G数・出率・BB・RB…」の表 → [{date, games, bb, rb}]（過去数日分）"""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue
        head = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
        if head[:1] != ["日付"] or "BB" not in head or "RB" not in head or "G数" not in head:
            continue
        ig, ib, ir = head.index("G数"), head.index("BB"), head.index("RB")
        for tr in rows[1:]:
            v = [c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
            if len(v) <= max(ig, ib, ir):
                continue
            m = _MD.search(v[0])
            if not m:
                continue
            y = int(m.group(1)) if m.group(1) else infer_year(int(m.group(2)), int(m.group(3)), m.group(4), today)
            if not y:
                continue
            g, bb, rb = _num(v[ig]), _num(v[ib]), _num(v[ir])
            if g is None or bb is None or rb is None:
                continue
            out.append({"date": f"{y:04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}", "games": g, "bb": bb, "rb": rb})
    return out


def est_diff(bb, rb, games, model: str = "", coef=None) -> int | None:
    """差枚が伏せられているとき（みんレポで ±1 や - と表示）は、機種ごとの
    BIG獲得枚数・REG獲得枚数・コイン持ちから計算する:
      差枚 ＝ BIG回数×BIG枚数 ＋ REG回数×REG枚数 − G数×(1Gあたりの実質消費)
    実質消費はコイン持ちを公表の機械割に合うように調整した値（specs.coin_use_per_game）。
    機種のスペックが未登録なら、実データで合わせた係数で見積もる。"""
    if bb is None or rb is None or games is None:
        return None
    from ..specs import coin_use_given, spec_for
    sp = spec_for(model)
    c = coin_use_given(sp, bb, rb, games) if sp.big_coins and sp.reg_coins else None
    if c:
        return int(round(bb * sp.big_coins + rb * sp.reg_coins - games * c))
    a, b, c = coef or DEFAULT_COEF
    return int(round(a * bb + b * rb + c * games))


def post_date(html: str, fallback: str) -> str:
    """日別ページの本当の日付（タイトルの月日 + 投稿日の年）"""
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text() if soup.title else ""
    m = _MD.search(title)
    p = _PUB.search(soup.get_text(" ", strip=True))
    if m and p:
        py, pm = int(p.group(1)), int(p.group(2))
        mm, dd = int(m.group(2)), int(m.group(3))
        y = py - 1 if mm > pm else py
        return f"{y:04d}-{mm:02d}-{dd:02d}"
    return fallback


# ---- BB/RB の推定（過去分用） ----
DEFAULT_COEF = (242.0, 131.0, -1.19)   # 差枚 ≒ a*BB + b*RB + c*G（WOの実データで校正した値）


def fit_payout_coef(df) -> tuple[float, float, float]:
    d = df.dropna(subset=["bb", "rb", "games", "diff"])
    d = d[d["games"] >= 1000]
    if len(d) < 200:
        return DEFAULT_COEF
    X = d[["bb", "rb", "games"]].to_numpy(float)
    y = d["diff"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = coef
    if not (150 < a < 350 and 60 < b < 200 and -3 < c < 0):
        return DEFAULT_COEF
    return float(a), float(b), float(c)


def estimate_bb_rb(diff, games, total, coef) -> tuple[int | None, int | None]:
    if diff is None or games is None or total is None:
        return None, None
    a, b, c = coef
    bb = (diff - c * games - b * total) / (a - b)
    bb = int(round(min(max(bb, 0), total)))
    return bb, total - bb


# ---- 取得処理 ----
class MinRepo:
    def __init__(self, db, fetcher: Fetcher, tag_url: str, keyword: str, interval: float = 6, parallel: int = 1):
        self.db, self.f, self.tag_url, self.kw, self.interval = db, fetcher, tag_url.rstrip("/") + "/", keyword, interval
        self.parallel = parallel

    def refresh_index(self, today: Date, all_pages: bool = False) -> list[tuple[str, str]]:
        html = self.f.get(self.tag_url, interval=self.interval)
        days, links, last = parse_tag_page(html, self.tag_url, today)
        if all_pages:
            for p in range(2, last + 1):
                h = self.f.get(f"{self.tag_url}page/{p}/", referer=self.tag_url, interval=self.interval)
                d2, l2, _ = parse_tag_page(h, self.tag_url, today)
                days += d2
                links += l2
        self.db.upsert_hall_days(days, "minrepo")
        self.db.upsert_index(links, "minrepo")
        return links

    def fetch_day(self, date: str, url: str, precise: bool, coef=DEFAULT_COEF) -> int:
        html = self.f.get(url, referer=self.tag_url, interval=self.interval)
        if len(html) < 3000:
            self._dump(html, url)
            raise RateLimited(f"日別ページの中身が空（{len(html)}文字） {url}")
        date = post_date(html, date)
        models, variety = parse_day_models(html, self.kw)
        if not models and not variety:
            self._dump(html, url)
        rows = []
        for model in models:
            kurl = f"{url.rstrip('/')}/?kishu={quote(model)}"
            units = parse_kishu_page(self.f.get(kurl, referer=url, interval=self.interval))
            for u in units:
                rows.append({"date": date, "unit": u["unit"], "model": model, "games": u["games"],
                             "diff": u["diff"], "total": u["total"]})
        return self._finish(date, url, rows, variety, precise, coef)

    def _dump(self, html: str, url: str) -> None:
        if getattr(self, "_dumped", False):
            return
        self._dumped = True
        if True:
            try:
                import json
                from pathlib import Path
                dp = Path(__file__).resolve().parents[2] / "data" / "debug" / "minrepo_day.json"
                dp.parent.mkdir(parents=True, exist_ok=True)
                dp.write_text(json.dumps(debug_dump(html, url), ensure_ascii=False, indent=1), encoding="utf-8")
            except Exception as e:
                log.warning("調査用の記録に失敗: %s", e)

    def _finish(self, date, url, rows, variety, precise, coef) -> int:
        for v in variety:
            rows.append({"date": date, "unit": v["unit"], "model": v["model"], "games": v["games"],
                         "diff": v["diff"], "total": None})
        if precise:
            out = []
            extra: dict[str, list[dict]] = {}
            done = self.db.fetched_dates("minrepo")
            pages = self.f.get_many([f"{url.rstrip('/')}/?num={r['unit']}" for r in rows], referer=url,
                                    parallel=self.parallel)
            for r, h in zip(rows, pages):
                info = parse_unit_page(h)
                try:   # 台番ページには過去数日分も載っているので、まだ取っていない日はここで一緒に取る
                    for hrow in parse_unit_history(h, Date.fromisoformat(date) + timedelta(days=1)):
                        if hrow["date"] < date and hrow["date"] not in done:
                            extra.setdefault(hrow["date"], []).append({
                                "date": hrow["date"], "unit": r["unit"], "model": r["model"],
                                "games": hrow["games"], "bb": hrow["bb"], "rb": hrow["rb"],
                                "diff": est_diff(hrow["bb"], hrow["rb"], hrow["games"], r["model"])})
                except Exception as e:
                    log.debug("過去分の読み取り失敗: %s", e)
                if info:
                    if not getattr(self, "_unit_dumped", False):
                        self._unit_dumped = True
                        try:
                            import json
                            from pathlib import Path
                            dp = Path(__file__).resolve().parents[2] / "data" / "debug" / "minrepo_unit.json"
                            dp.parent.mkdir(parents=True, exist_ok=True)
                            soup = BeautifulSoup(h, "lxml")
                            tabs = [[[c.get_text(strip=True) for c in tr.find_all(["th", "td"])] for tr in t.find_all("tr")[:40]]
                                    for t in soup.find_all("table")[:6]]
                            dp.write_text(json.dumps({"url": url, "unit": r["unit"], "row": {k: v for k, v in r.items()},
                                                      "info": info, "tables": tabs}, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
                        except Exception as e:
                            log.warning("調査用の記録に失敗: %s", e)
                    games = info.get("games") or r.get("games")
                    d2 = info.get("diff")
                    if d2 is None or abs(d2) <= 1:     # 差枚が伏せられている店
                        d2 = est_diff(info.get("bb"), info.get("rb"), games, r["model"])
                    out.append({**r, "bb": info.get("bb"), "rb": info.get("rb"), "diff": d2, "games": games})
            n = self.db.upsert_units(out, "minrepo")
            # 一緒に取れた過去日: その日の台数がそろっていれば取得済みにする
            for d, ex in extra.items():
                if len(ex) >= max(1, int(len(out) * 0.9)):
                    self.db.upsert_units(ex, "minrepo")
                    self.db.log_fetch(d, "minrepo", "ok", len(ex))
                    self.extra_days = getattr(self, "extra_days", []) + [d]
        else:
            out = []
            for r in rows:
                bb, rb = estimate_bb_rb(r["diff"], r["games"], r["total"], coef)
                if bb is not None:
                    out.append({**r, "bb": bb, "rb": rb})
            n = self.db.upsert_units(out, "minrepo_est", est=True)
        self.db.log_fetch(date, "minrepo", "ok" if out else "empty", n)
        return n

    def fill_recent(self, today: Date, after: str | None, max_days: int = 5) -> int:
        """アナスロの最新日より新しい日がみんレポにあれば正確に取得。"""
        if not after:
            return 0
        links = self.refresh_index(today)
        done = self.db.fetched_dates("minrepo")
        todo = [(d, u) for d, u in sorted(links, reverse=True) if (after is None or d > after) and d not in done]
        n = 0
        for d, u in todo[:max_days]:
            try:
                n += self.fetch_day(d, u, precise=True)
            except RateLimited:
                log.warning("みんレポ: アクセス制限。今回は中断します")
                break
            except Exception as e:
                log.warning("みんレポ: %s の取得失敗: %s", d, e)
        return n

    def backfill_history(self, today: Date, before: str, since: str, coef, deadline: float | None = None,
                         max_days: int = 2000) -> int:
        """アナスロより古い日を推定値で取得。"""
        links = self.refresh_index(today, all_pages=True)
        done = self.db.fetched_dates("minrepo")
        todo = [(d, u) for d, u in sorted(links, reverse=True) if since <= d < before and d not in done]
        log.info("みんレポ過去分: 対象 %d日", len(todo))
        n = 0
        for d, u in todo[:max_days]:
            if deadline and time.time() > deadline:
                break
            try:
                n += self.fetch_day(d, u, precise=False, coef=coef)
                log.info("みんレポ過去分: %s 取得", d)
            except RateLimited:
                log.warning("みんレポ: アクセス制限。今回は中断します")
                break
            except Exception as e:
                log.warning("みんレポ: %s の取得失敗: %s", d, e)
        return n
