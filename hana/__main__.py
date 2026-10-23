"""使い方（プロジェクトのフォルダで）:

  python -m hana daily              毎朝用: 新しいデータを取得して今日の予想ページを作る
  python -m hana predict [--date 2026-10-11] [--open]
  python -m hana update [--max-pages 8]
  python -m hana backfill [--hours 8]   過去データを時間をかけて取得（アクセス制限に配慮してゆっくり）
  python -m hana backtest [--days 90] [--lag 1]
  python -m hana minrepo-history [--since 2024-01-01] [--hours 3]   みんレポの古いデータ（任意）
  python -m hana status
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import webbrowser
from datetime import date as Date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as BT
from . import config as C
from . import model as M
from . import report as R
from .db import DB
from .http import Fetcher, RateLimited
from .sources import anaslo
from .sources import minrepo as MR

JST = timezone(timedelta(hours=9))
log = logging.getLogger("hana")


def setup_logging(root: Path) -> None:
    (root / "logs").mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(root / "logs" / "hana.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg = logging.getLogger("hana")
    lg.setLevel(logging.INFO)
    lg.handlers[:] = [fh, sh]
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class FileLock:
    """取得処理が同時に2つ動かないようにするロック（PC再起動やプロセス終了で自動解除）。"""
    def __init__(self, path: Path):
        self.path = path
        self.f = None

    def acquire(self) -> bool:
        import os
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                self.f.seek(0)
                msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False


def today_jst() -> Date:
    return datetime.now(JST).date()


def make_fetcher(cfg: dict) -> Fetcher:
    f = cfg["fetch"]
    return Fetcher(interval_sec=f.get("interval_sec", 75), jitter_sec=f.get("jitter_sec", 30),
                   engine=f.get("engine", "requests"))


def write_status(root: Path, **kw) -> None:
    p = root / "data" / "status.json"
    try:
        cur = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        cur = {}
    cur.update(kw)
    p.write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_update(cfg, db, max_pages=None, deadline=None) -> dict:
    stats = {}
    if not cfg["fetch"].get("anaslo_direct", True):
        return cmd_minrepo_recent(cfg, db, stats)
    f = make_fetcher(cfg)
    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    try:
        stats = anaslo.update(db, f, cfg["hall"]["anaslo_list_url"],
                              max_pages=max_pages or cfg["fetch"].get("daily_max_pages", 8),
                              backoff_min=cfg["fetch"].get("backoff_min", 20), deadline=deadline)
        ok = stats.get("new_days", 0) > 0 or not stats.get("rate_limited")
        write_status(C.ROOT, anaslo_checked=now, anaslo_result="ok" if ok else "limited",
                     anaslo_message="" if ok else "アクセス制限で一部取得できず（次回続きから）",
                     **({"anaslo_ok": now} if ok else {}))
    except RateLimited as e:
        log.warning("アナスロ: 一覧ページもアクセス制限中。次回の実行で取得します")
        stats["rate_limited"] = True
        write_status(C.ROOT, anaslo_checked=now, anaslo_result="blocked", anaslo_message=f"アクセスを拒否されました（{e}）")
    except Exception as e:
        log.error("アナスロの取得でエラー: %s", e)
        msg = str(e)
        if any(k in msg for k in ("ConnectionError", "ProxyError", "Max retries", "timed out", "NameResolution")):
            msg = "サイトに接続できませんでした"
        write_status(C.ROOT, anaslo_checked=now, anaslo_result="error", anaslo_message=msg[:120])
    f.close()
    return cmd_minrepo_recent(cfg, db, stats)


def cmd_minrepo_recent(cfg, db, stats: dict) -> dict:
    # みんレポ: アナスロの最新日より新しい日があれば補完
    try:
        cov = db.coverage()
        mf = Fetcher(interval_sec=cfg["fetch"].get("minrepo_interval_sec", 8), jitter_sec=3)
        mr = MR.MinRepo(db, mf, cfg["hall"]["minrepo_tag_url"], cfg["target"].get("keyword", "ハナハナ"),
                        interval=cfg["fetch"].get("minrepo_interval_sec", 8))
        n = mr.fill_recent(today_jst(), after=cov.get("unit_last"))
        if n:
            log.info("みんレポで %d 台日を補完", n)
    except Exception as e:
        log.warning("みんレポの確認でエラー（無視して続行）: %s", e)
    return stats


def cmd_minrepo_sync(cfg, db, root: Path, minutes: float | None = None) -> dict:
    """みんレポから、まだ取っていない日を新しい順に正確な値（BB/RB回数）で取得する。
    1回の実行で minutes 分だけ取り、残りは次回の実行で続きから。"""
    mc = cfg.get("minrepo", {})
    minutes = float(mc.get("minutes_per_run", 18) if minutes is None else minutes)
    since = str(mc.get("since", "2025-01-01"))
    deadline = time.time() + minutes * 60
    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    interval = cfg["fetch"].get("minrepo_interval_sec", 2)
    f = Fetcher(interval_sec=interval, jitter_sec=1, engine=cfg["fetch"].get("minrepo_engine", "playwright"))
    mr = MR.MinRepo(db, f, cfg["hall"]["minrepo_tag_url"], cfg["target"].get("keyword", "ハナハナ"),
                    interval=interval, parallel=int(cfg["fetch"].get("minrepo_parallel", 3)))
    got, blocked, err = [], False, ""
    try:
        full = not db.index("minrepo") or datetime.now(JST).hour == 3
        links = mr.refresh_index(today_jst(), all_pages=full)
        if not full:
            links = list(db.index("minrepo").items())
        done = db.fetched_dates("minrepo")
        todo = [(d, u) for d, u in sorted(links, reverse=True) if d >= since and d not in done]
        log.info("みんレポ: 未取得 %d日（%s 以降）", len(todo), since)
        for d, u in todo:
            if time.time() > deadline:
                break
            if d in db.fetched_dates("minrepo"):   # 前の日の台ページから一緒に取れた日
                continue
            try:
                n = mr.fetch_day(d, u, precise=True)
                got.append(d)
                more = getattr(mr, "extra_days", [])
                got += more
                mr.extra_days = []
                log.info("みんレポ: %s ハナハナ %d台%s", d, n, f"（ほか過去{len(more)}日分も取得）" if more else "")
            except RateLimited:
                blocked = True
                log.warning("みんレポ: アクセス制限。続きは次回")
                break
            except Exception as e:
                log.warning("みんレポ: %s の取得失敗: %s", d, e)
            db.export_csv(root / "data", keyword=cfg["target"].get("keyword", "ハナハナ"))
        remain = len([d for d, _ in todo if d not in db.fetched_dates("minrepo")])
    except RateLimited as e:
        blocked, remain, err = True, None, str(e)
    except Exception as e:
        blocked, remain, err = False, None, str(e)[:120]
        log.error("みんレポの取得でエラー: %s", e)
    f.close()
    result = "ok" if got or (not blocked and not err) else ("blocked" if blocked else "error")
    msg = f"{len(got)}日分を取得" + (f"・残り{remain}日" if remain else "") + (f"（{err}）" if err and not got else "")
    kw = {"mode": "minrepo", "minrepo_last": now, "minrepo_result": result, "minrepo_message": msg}
    if got:
        kw["minrepo_ok"] = now
    write_status(root, **kw)
    return {"days": len(got), "remain": remain, "blocked": blocked}


def cmd_inbox(cfg, db, root: Path) -> dict:
    """iPhone から届いたデータを取り込み、次に取りに行くページの一覧を更新する。"""
    from .sources import gh_inbox, phone
    st = phone.process(db, root)
    try:
        num = gh_inbox.process(db, st, root)
        if num:
            write_status(root, inbox_issue=num)
    except Exception as e:
        log.warning("受信箱(Issue)の確認でエラー: %s", e)
    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    if st["files"]:
        result = "ok" if st["days"] else ("blocked" if st["blocked"] else "nodata")
        kw = {"phone_last": now, "phone_result": result,
              "phone_message": f"{len(set(st['dates'])) or st['days']}日分を受信"
                               + (("（" + "・".join(f"{int(d[5:7])}/{int(d[8:10])}" for d in sorted(set(st["dates"]), reverse=True)) + "）")
                                  if st.get("dates") else "")
                               + (f"・空ページ{st['blocked']}件" if st["blocked"] else "")}
        if st["days"]:
            kw["phone_ok"] = now
        write_status(root, **kw)
        log.info("iPhoneからの受信: %s", st)
    write_next(cfg, db, root)
    return st


def write_next(cfg, db, root: Path) -> list[str]:
    from .sources import phone
    tmpl = cfg["hall"]["anaslo_day_url"]
    n = int(cfg.get("phone", {}).get("pages_per_run", 1))
    hp = root / "data" / "handout.json"
    try:
        handout = json.loads(hp.read_text(encoding="utf-8")) if hp.exists() else {}
    except Exception:
        handout = {}
    urls = phone.next_urls(db, today_jst(), tmpl, n=n, handout=handout)
    # 手動の「まとめて取得」用（先頭は自動実行と同じ日付。失敗回数の記録は上の1回分だけ）
    many_n = int(cfg.get("phone", {}).get("pages_manual", 6))
    many = phone.next_urls(db, today_jst(), tmpl, n=many_n, handout=json.loads(json.dumps(handout)))
    many_txt = "\n".join(many)
    (root / "data" / "next_many.txt").write_text(many_txt, encoding="utf-8")
    hp.write_text(json.dumps(dict(sorted(handout.items())), ensure_ascii=False, indent=1), encoding="utf-8")
    txt = "\n".join(urls)          # 末尾に改行を付けない（iPhone側で空の行がエラーになるため）
    (root / "data" / "next.txt").write_text(txt, encoding="utf-8")
    (root / "output").mkdir(exist_ok=True)
    (root / "output" / "next.txt").write_text(txt, encoding="utf-8")
    log.info("iPhoneが次に取りに行くページ: %d件", len(urls))
    return urls


def cmd_backfill(cfg, db, hours: float, on_progress=None) -> None:
    deadline = time.time() + hours * 3600
    while time.time() < deadline:
        st = cmd_update(cfg, db, max_pages=60, deadline=deadline)
        log.info("取得状況: %s", st)
        if on_progress:
            on_progress()
        if st.get("remaining", 0) <= 0:
            log.info("アナスロの取得可能な過去データはすべて取得しました")
            break
        if st.get("rate_limited"):
            wait = cfg["fetch"].get("backoff_min", 20) * 3
            log.info("%d分休んでから続けます", wait)
            time.sleep(min(wait * 60, max(0, deadline - time.time())))


def load_backtest(root: Path) -> dict | None:
    p = root / "data" / "backtest.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def need_backtest(root: Path) -> bool:
    bt = load_backtest(root)
    if bt is None or not bt.get("ok") or bt.get("goal") != C.model_settings(C.load()).goal:
        return True
    return today_jst().weekday() == 0 and bt.get("when") != str(today_jst())


def load_factors(root: Path) -> None:
    """この店のデータで選んだ要因（data/factors.json）を読み込む。まだ無ければ初期値のまま。"""
    p = root / "data" / "factors.json"
    try:
        f = json.loads(p.read_text(encoding="utf-8"))
        if f.get("ok") and f.get("active"):
            M.use_factors(f["active"], {"source": "selected", **f})
    except Exception:
        pass


def need_selection(root: Path, n_days: int) -> bool:
    p = root / "data" / "factors.json"
    try:
        f = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return True
    if not f.get("ok") or f.get("goal") != C.model_settings(C.load()).goal:
        return True       # 予想の狙い（goal）を変えたら選び直す
    when = Date.fromisoformat(f.get("when", "2000-01-01"))
    # 週1回、またはデータが2割以上増えたら選び直す
    return (today_jst() - when).days >= 7 or n_days >= int(f.get("n_days", 0) * 1.2) + 1


def cmd_select(cfg, db, root: Path, force: bool = False) -> dict | None:
    units, hall = db.units(cfg["target"].get("keyword", "ハナハナ")), db.hall_days()
    n_days = units["date"].nunique() if len(units) else 0
    if not force and not need_selection(root, n_days):
        return None
    res = BT.select_factors(units, hall, C.model_settings(cfg))
    res["when"] = str(today_jst())
    res["goal"] = C.model_settings(cfg).goal
    (root / "data" / "factors.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    if res.get("ok"):
        log.info("この店で使う要因を選び直しました: %s", " / ".join(s["label"] for s in res["steps"]) or "なし")
    else:
        log.info("要因の選び直し: %s", res.get("reason"))
    load_factors(root)
    return res


def maybe_fit_prior(cfg, units, root: Path) -> None:
    from . import quitrule as Q
    p = root / "data" / "setting_prior.json"
    try:
        when = Date.fromisoformat(json.loads(p.read_text(encoding="utf-8")).get("when", "2000-01-01"))
    except Exception:
        when = Date(2000, 1, 1)
    if (today_jst() - when).days >= 7:
        try:
            r = Q.fit_setting_prior(units, root, cfg["target"].get("keyword", "ハナハナ"))
            log.info("この店の設定配分を推定し直しました: %s", r["weights"])
        except Exception as e:
            log.warning("設定配分の推定に失敗（前回の値のまま）: %s", e)


def run_system(cfg, units, root: Path, target: Date) -> dict:
    """予想システム: 全台の点数、2年分の答え合わせ（週1回やり直し）、店の癖。"""
    from . import sysmodel as SM
    kw = cfg["target"].get("keyword", "ハナハナ")
    ev = tuple(cfg["hall"].get("event_digits", [0, 5]))
    d = SM.build(units, kw, ev, target=pd.Timestamp(target))
    pred = SM.predict_units(d, target)
    hp, sp = root / "data" / "system_history.csv", root / "data" / "system.json"
    try:
        summ = json.loads(sp.read_text(encoding="utf-8"))
    except Exception:
        summ = {}
    n_days = int(d.loc[d["obs"], "date"].nunique())
    when = summ.get("when", "2000-01-01")
    if (not summ or summ.get("method") != SM.MAIN_METHOD or not hp.exists()
            or (today_jst() - Date.fromisoformat(when)).days >= 7):
        log.info("予想システムの答え合わせを作り直します（過去の各日を、その日より前のデータだけで予想し直す）")
        h = SM.backtest(d)
        h.to_csv(hp, index=False)
        summ = {"when": str(today_jst()), "method": SM.MAIN_METHOD, "n_days": n_days, **SM.history_summary(h)}
        sp.write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    try:
        research = json.loads((root / "data" / "research.json").read_text(encoding="utf-8"))
    except Exception:
        research = {}
    if summ.get("buckets") and "pay_s" not in summ["buckets"][0]:
        summ["buckets"] = SM.smooth_buckets(summ["buckets"])
    quit_info = {}
    try:   # 続けるか・移るか・やめるかの計算機（今日の日の強さも見込みに入れる）
        from . import daystrength as DS
        from . import quitrule as Q
        ds = DS.estimate(d[d["obs"]], target, ev)
        delta = float(ds["today"]["delta"]) if ds else 0.0
        params = Q.page_params(units, summ, delta, kw)
        try:
            qr = json.loads((root / "data" / "quit_research.json").read_text(encoding="utf-8"))
        except Exception:
            qr = {}
        quit_info = {"params": params, "table": Q.quick_table(params), "research": qr, "margin": Q.MOVE_MARGIN}
    except Exception as e:
        log.warning("撤退判断の準備に失敗: %s", e)
    return {"pred": pred, "summary": summ, "research": research, "habits": SM.habits(d), "quit": quit_info}


def load_methods(root: Path) -> dict:
    try:
        return json.loads((root / "data" / "methods.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def cmd_compare(cfg, db, root: Path, force: bool = False) -> dict | None:
    """予想方法の比較（週1回・データが2割増えたとき・狙いを変えたとき）。一番よい方法を data/methods.json に残す。"""
    from . import ml
    units, hall = db.units(cfg["target"].get("keyword", "ハナハナ")), db.hall_days()
    n_days = units["date"].nunique() if len(units) else 0
    s = C.model_settings(cfg)
    old = load_methods(root)
    if not force and old.get("ok") and old.get("kind") == "all":
        when = Date.fromisoformat(old.get("when", "2000-01-01"))
        if (today_jst() - when).days < 7 and n_days < int(old.get("n_days", 0) * 1.2) + 1:
            return None
    res = ml.compare_all(units, s)
    res["when"] = str(today_jst())
    res["kind"] = "all"
    (root / "data" / "methods.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if res.get("ok") and res.get("best"):
        b = res["summary"][res["best"]]
        log.info("予想方法を比較: 一番当たるのは「%s」を「%s」で予想（2台比べて当たる割合 %.1f%%）",
                 b["target_label"], b["label"], b["acc"] * 100)
    return res


def run_checks(cfg, db, root: Path) -> None:
    """休業日の候補の記録、要因の選び直し（週1回・データが2割増えたとき）、予想の検証（週1回）。"""
    try:
        detect_closed_days(cfg, db, root)
    except Exception as e:
        log.warning("休業日の確認でエラー: %s", e)
    if cfg["predict"].get("method", "system") == "system":
        return     # 予想システムの答え合わせは run_system の中で週1回作り直す
    try:
        cmd_select(cfg, db, root)
    except Exception as e:
        log.warning("要因の選び直しでエラー（前回の選択のまま続行）: %s", e)
    try:
        cmd_compare(cfg, db, root)
    except Exception as e:
        log.warning("予想方法の比較でエラー（前回の結果のまま続行）: %s", e)
    if need_backtest(root):
        try:
            cmd_backtest(cfg, db, root)
        except Exception as e:
            log.warning("検証でエラー（予想は続行）: %s", e)


def cmd_backtest(cfg, db, root: Path, days=90, lag=None) -> dict:
    units, hall = db.units(cfg["target"].get("keyword", "ハナハナ")), db.hall_days()
    try:
        cmd_select(cfg, db, root)
    except Exception as e:
        log.warning("要因の選び直しでエラー（前回の選択のまま続行）: %s", e)
    if lag is None:
        lag = estimate_lag(db)
    res = BT.run(units, hall, C.model_settings(cfg), days=days, lag=lag)
    import numpy as np

    def plain(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return float(v)
        return v
    out = {k: plain(v) for k, v in res.items() if k != "table"}
    out["when"] = f"{datetime.now():%Y-%m-%d}"
    out["goal"] = C.model_settings(cfg).goal
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "backtest.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    if "table" in res:
        res["table"].to_csv(root / "data" / "backtest_days.csv", index=False, encoding="utf-8-sig")
    return out


def estimate_lag(db) -> int:
    """データ公開の遅れ（日）。最近の取得記録から推定。"""
    cov = db.coverage()
    if not cov.get("unit_last"):
        return 1
    last = Date.fromisoformat(cov["unit_last"])
    return max(1, min(7, (today_jst() - last).days))


def cmd_predict(cfg, db, root: Path, target: Date, open_browser=False) -> Path | None:
    units, hall = db.units(cfg["target"].get("keyword", "ハナハナ")), db.hall_days()
    if units.empty:
        log.error("データがありません。先に update / backfill を実行してください")
        out = root / "output"
        out.mkdir(exist_ok=True)
        name = cfg["hall"].get("name", "")
        page = ("<!doctype html><html lang='ja'><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                f"<title>{name} ハナハナ予想</title></head>"
                "<body style='font-family:-apple-system,sans-serif;padding:24px;line-height:1.7'>"
                f"<h1 style='font-size:20px'>{name} ハナハナ予想</h1>"
                "<p>過去データを集めている最中です。データが届きしだい、ここに予想ランキングが表示されます。</p>"
                f"<p style='color:#666'>{status_line(root) or 'iPhoneからの受信はまだありません'}</p></body></html>")
        (out / "index.html").write_text(page, encoding="utf-8")
        return None
    s = C.model_settings(cfg)
    if cfg["predict"].get("method", "system") == "system":
        maybe_fit_prior(cfg, units, root)      # この店の設定配分（差枚の計算と撤退判断に使う）を週1回推定し直す
    m = M.train(units, hall, s)
    pred = M.predict(m, target)
    if cfg["predict"].get("method", "system") == "system":
        try:   # 2026/10/22 の比較で一番成績のよかった予想システム（sysmodel.py）で並べ直す
            info = run_system(cfg, units, root, target)
            sp = info.pop("pred", None)
            if sp is not None:
                pred = pred.drop(columns=[c for c in ("why",) if c in pred.columns]).merge(
                    sp[["unit", "sys_score", "why"]], on="unit", how="left")
                r = pred["sys_score"].rank(method="first")
                pred["score"] = ((r - 0.5) / max(int(r.notna().sum()), 1)).fillna(0.0)
                pred["why"] = pred["why"].apply(lambda v: v if isinstance(v, list) else [])
                pred = pred.sort_values("score", ascending=False).reset_index(drop=True)
                pred["rank"] = np.arange(1, len(pred) + 1)
                pred.attrs["system"] = info
        except Exception as e:
            log.warning("予想システムでの予想に失敗（これまでの方法で表示）: %s", e)
    pred_dir = root / "data" / "predictions"
    try:   # 台ごと・日ごとの判定設定を記録
        j = m.hist[["date", "unit", "model", "games", "bb", "rb", "diff", "gassan", "reg_den", "judge", "p_high"]].copy()
        j["date"] = j["date"].dt.strftime("%Y-%m-%d")
        j["gassan"] = j["gassan"].round(1)
        j["reg_den"] = j["reg_den"].round(1)
        j["judge"] = j["judge"].astype("Int64")
        j["p_high"] = j["p_high"].round(3)
        j.columns = ["日付", "台番", "機種", "G数", "BIG", "REG", "差枚", "合算分母", "REG分母", "判定設定", "高設定確率"]
        j.sort_values(["日付", "台番"]).to_csv(root / "data" / "daily_settings.csv", index=False, encoding="utf-8")
    except Exception as e:
        log.warning("判定設定の記録に失敗: %s", e)
    try:   # 動作環境の記録（クラウドと手元で結果を比べるため）
        import platform
        import numpy
        import pandas as pd
        envp = root / "data" / "debug" / "env.json"
        envp.parent.mkdir(parents=True, exist_ok=True)
        envp.write_text(json.dumps({"python": platform.python_version(), "pandas": pd.__version__,
                                    "numpy": numpy.__version__, "target": str(target),
                                    "top": [[int(r.unit), round(float(r.score), 4)] for r in pred.head(5).itertuples()]},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        log.warning("環境の記録に失敗: %s", e)
    path = R.build(m, pred, target, cfg["hall"].get("name", ""), cfg["predict"].get("top_n", 10),
                   load_backtest(root), db.coverage(), root / "output",
                   extra_warns=variation_note(units, s, system=bool(pred.attrs.get("system"))) + closed_notes(s, units, target) + status_warnings(root),
                   pred_dir=pred_dir, info_line=status_line(root))
    if cfg.get("storage") == "csv":   # 答え合わせ用に予想を保存
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred[["rank", "unit", "score"]].round(4).to_csv(pred_dir / f"{target:%Y-%m-%d}.csv", index=False)
    log.info("予想ページを作成: %s", path)
    top = pred.head(5)
    log.info("上位: %s", "  ".join(f"{int(r.unit)}({r.score*100:.1f}%)" for r in top.itertuples()))
    if open_browser:
        webbrowser.open((root / "output" / "latest.html").resolve().as_uri())
    return path


def variation_note(units, s, system: bool = False) -> list[str]:
    """店全体の設定のばらつき（高設定を入れているか）のお知らせ。"""
    from .estimate import variation_test
    from .specs import spec_for
    d = units[units["model"].str.contains(s.keyword, na=False)].copy()
    if d.empty:
        return []
    d["date"] = pd.to_datetime(d["date"])
    v = variation_test(d, s.judge_min_games)
    if not v.get("ok"):
        return []
    eq = []
    for m, x in v["models"].items():
        sp = spec_for(m)
        gs = [(i + 1, 1 / (1 / b + 1 / r)) for i, (b, r) in enumerate(zip(sp.bb, sp.rb)) if b]
        near = min(gs, key=lambda t: abs(t[1] - x["gassan"]))[0]
        eq.append(f"{m.split('‐')[0].split('-')[0]} 合算1/{x['gassan']:.0f}（設定{near}相当）")
    head = f"この店のハナハナ全体（{v['days']}日・{v['n']:,}台日）: " + "・".join(eq) + "。"
    if v["lr"] < 10:
        tail = ("ただし小さな差はあり、2年分の答え合わせでは予想上位の台の出率が全台平均より約1%高くなっています（下の「2年間の答え合わせ」）。"
                if system else "ランキングは参考程度に見てください。")
        return [head + f"台ごと・日ごとの数字のばらつきが「全台同じ設定」とした場合の運の範囲に収まっていて"
                       f"（判定値 {v['lr']:.0f}／10未満）、1日の数字で分かるほどの高設定（1日に設定6が3台・設定4が5台以上など）は"
                       "入っていないと考えられます。" + tail]
    if v["lr"] < 30:
        return [head + f"設定のばらつきは少しだけ見えます（判定値 {v['lr']:.0f}）。高設定はごく少数と考えられます。"]
    return [head + f"設定のばらつきがはっきり見えます（判定値 {v['lr']:.0f}）。高設定を入れている店です。"]


def detect_closed_days(cfg, db, root: Path) -> list[str]:
    """みんレポに日別ページが無い日（前後の日はある）を休業日の候補として data/closed_days.txt に記録する。"""
    idx = sorted(db.index("minrepo"))
    since = str(cfg.get("minrepo", {}).get("since", "2025-01-01"))
    idx = [d for d in idx if d >= since]
    if len(idx) < 30:
        return []
    have = set(idx)
    gaps = [d.date().isoformat() for d in pd.date_range(idx[0], idx[-1]) if d.date().isoformat() not in have]
    p = root / "data" / "closed_days.txt"
    old = set(p.read_text(encoding="utf-8").split()) if p.exists() else set()
    new = sorted(old | set(gaps))
    if new != sorted(old):
        p.write_text("\n".join(new) + "\n", encoding="utf-8")
        log.info("休業日の候補（みんレポに掲載なし）: %s", ", ".join(sorted(set(gaps) - old)))
    return new


def closed_notes(s, units, target: Date) -> list[str]:
    """休業日まわりのお知らせ（予想ページの上に出す）。"""
    out = []
    closed = set(s.closed_days)
    t = target.isoformat()
    if t in closed:
        out.append(f"本日（{target.month}/{target.day}）は休業日として登録されています。予想は参考表示です。")
    prev = M.prev_open_day(target, 1, s).date()
    skipped = [d for d in sorted(closed) if prev.isoformat() < d < t]
    have = set(units["date"].astype(str)) if len(units) else set()
    if skipped:
        names = "・".join(f"{int(d[5:7])}/{int(d[8:10])}" for d in skipped)
        out.append(f"{names} は休業日のため、{prev.month}/{prev.day} のデータを「前日」として予想しています。")
    if prev.isoformat() not in have:
        y = target - timedelta(days=1)
        if y.isoformat() not in closed:
            out.append(f"{prev.month}/{prev.day} が休業日だった場合は、config.yaml の closed_days に"
                       f" {prev.isoformat()} を追加すると、その前の営業日を「前日」として予想します。")
    return out


def status_line(root: Path) -> str:
    p = root / "data" / "status.json"
    try:
        st = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if st.get("mode") == "minrepo":
        return f"みんレポの最終取得: {st.get('minrepo_last', 'まだありません')}" + (f"（{st['minrepo_message']}）" if st.get('minrepo_message') else "")
    if st.get("mode") == "phone":
        return f"iPhoneからの最終受信: {st.get('phone_last', 'まだありません')}" + (f"（{st['phone_message']}）" if st.get('phone_message') else "")
    return ""


def status_warnings(root: Path) -> list[str]:
    p = root / "data" / "status.json"
    if not p.exists():
        return []
    try:
        st = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    out = []
    if st.get("mode") == "minrepo":
        if st.get("minrepo_result") in ("blocked", "error"):
            out.append(f"みんレポから取得できませんでした（{st.get('minrepo_last')}：{st.get('minrepo_message', '')}）。"
                       f"最後に取得できたのは {st.get('minrepo_ok', 'なし')} です。次回の実行で自動的に再試行します。")
        return out
    if st.get("mode") == "phone":
        last = st.get("phone_last")
        if not last:
            out.append("iPhoneからのデータ送信がまだありません。設定ガイドの手順でショートカットを設定してください。")
        else:
            try:
                age = datetime.now(JST).replace(tzinfo=None) - datetime.strptime(last, "%Y-%m-%d %H:%M")
                if age > timedelta(hours=36):
                    out.append(f"iPhoneからのデータ送信が止まっています（最後の受信 {last}）。ショートカットのオートメーションを確認してください。")
            except ValueError:
                pass
            if st.get("phone_result") == "blocked":
                out.append(f"最後の受信（{last}）ではアナスロのページが空でした。アクセス制限の可能性があります（自動で再試行します）。")
        return out
    if st.get("anaslo_result") in ("blocked", "error"):
        last_ok = st.get("anaslo_ok", "なし")
        out.append(f"アナスロからデータを取得できていません（最終確認 {st.get('anaslo_checked')}：{st.get('anaslo_message', '')}）。"
                   f"最後に取得できたのは {last_ok} です。取得できるまでは手元のデータで予想しています。")
    return out


def cmd_import_seed(db, root: Path) -> None:
    import re
    seed = root / "data" / "seed"
    p = seed / "anaslo_hana.tsv"
    if p.exists():
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            d, model, unit, g, diff, bb, rb = line.split("\t")
            rows.append({"date": d, "unit": int(unit), "model": model, "games": anaslo.to_int(g),
                         "diff": anaslo.to_int(diff), "bb": anaslo.to_int(bb), "rb": anaslo.to_int(rb)})
        db.upsert_units(rows, "seed")
        for d in sorted({r["date"] for r in rows}):
            db.log_fetch(d, "anaslo", "ok", sum(1 for r in rows if r["date"] == d))
        log.info("初期データ %d 台日を取り込みました", len(rows))
    p = seed / "anaslo_hall_days.csv"
    if p.exists():
        rx = re.compile(r"^(\d{4}/\d{2}/\d{2}),(–|[+\-][\d,]+?),(–|[+\-][\d,]+?),([\d,]+)$")
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            m = rx.match(line.strip())
            if m:
                rows.append({"date": m.group(1).replace("/", "-"), "total_diff": anaslo.to_int(m.group(2)),
                             "avg_diff": anaslo.to_int(m.group(3)), "avg_games": anaslo.to_int(m.group(4))})
        db.upsert_hall_days(rows, "anaslo")
        log.info("店全体の日別データ %d 日を取り込みました", len(rows))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="hana", description="ハナハナ予想")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("daily")
    p = sub.add_parser("predict")
    p.add_argument("--date")
    p.add_argument("--open", action="store_true")
    p = sub.add_parser("update")
    p.add_argument("--max-pages", type=int)
    p = sub.add_parser("backfill")
    p.add_argument("--hours", type=float, default=8)
    p.add_argument("--max-minutes", type=float)
    p = sub.add_parser("backtest")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--lag", type=int)
    p = sub.add_parser("minrepo-history")
    p.add_argument("--since")
    p.add_argument("--hours", type=float)
    sub.add_parser("status")
    sub.add_parser("select")
    sub.add_parser("page")
    sub.add_parser("compare")
    sub.add_parser("import-seed")
    sub.add_parser("diag")
    sub.add_parser("inbox")
    a = ap.parse_args(argv)

    root = C.ROOT
    setup_logging(root)
    cfg = C.load()
    load_factors(root)
    csv_mode = cfg.get("storage") == "csv"
    db_path = root / "data" / "hana.db"
    if csv_mode and db_path.exists():
        db_path.unlink()          # CSV が正。毎回CSVから作り直す
    db = DB(db_path)
    if csv_mode:
        n = db.import_csv(root / "data")
        if n == 0:
            cmd_import_seed(db, root)
        log.info("保存データを読み込み: %s", db.coverage())

    lock = FileLock(root / "data" / "fetch.lock")
    fetch_ok = True
    if a.cmd in ("update", "backfill", "daily", "minrepo-history"):
        fetch_ok = lock.acquire()
        if not fetch_ok:
            log.info("別の取得処理が実行中のため、今回はデータ取得をスキップします")

    if a.cmd == "update":
        if fetch_ok:
            log.info("結果: %s", cmd_update(cfg, db, a.max_pages))
    elif a.cmd == "backfill":
        if not fetch_ok:
            return 0
        hours = a.max_minutes / 60 if a.max_minutes else a.hours
        save = (lambda: db.export_csv(root / "data", keyword=cfg["target"].get("keyword", "ハナハナ"))) if csv_mode else None
        cmd_backfill(cfg, db, hours, on_progress=save)
        try:
            cmd_backtest(cfg, db, root)
        except Exception as e:
            log.warning("検証でエラー（予想は続行）: %s", e)
        cmd_predict(cfg, db, root, today_jst())
    elif a.cmd == "predict":
        target = Date.fromisoformat(a.date) if a.date else today_jst()
        cmd_predict(cfg, db, root, target, a.open or cfg["predict"].get("open_browser", False))
    elif a.cmd == "backtest":
        r = cmd_backtest(cfg, db, root, a.days, a.lag)
        log.info("検証結果: %s", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    elif a.cmd == "inbox":
        cmd_inbox(cfg, db, root)
        cmd_minrepo_sync(cfg, db, root)
        run_checks(cfg, db, root)
        cmd_predict(cfg, db, root, today_jst())
        write_next(cfg, db, root)
    elif a.cmd == "daily":
        cmd_inbox(cfg, db, root)
        cmd_minrepo_sync(cfg, db, root)
        run_checks(cfg, db, root)
        cmd_predict(cfg, db, root, today_jst(), cfg["predict"].get("open_browser", False))
        write_next(cfg, db, root)
    elif a.cmd == "minrepo-history":
        if not fetch_ok:
            return 0
        cov = db.coverage()
        units = db.units(cfg["target"].get("keyword", "ハナハナ"))
        coef = MR.fit_payout_coef(units[units["source"].isin(["anaslo", "seed", "minrepo"])])
        log.info("BB/RB推定の係数: %s", coef)
        allu = db.units()
        an = allu[allu["source"] == "anaslo"]["date"]
        first = an.min() if len(an) else (cov.get("unit_first") or "2099-01-01")
        f = Fetcher(interval_sec=cfg["fetch"].get("minrepo_interval_sec", 8), jitter_sec=4)
        mr = MR.MinRepo(db, f, cfg["hall"]["minrepo_tag_url"], cfg["target"].get("keyword", "ハナハナ"),
                        interval=cfg["fetch"].get("minrepo_interval_sec", 8))
        hcfg = cfg.get("history", {})
        since = a.since or str(hcfg.get("since", "2024-01-01"))
        hours = a.hours if a.hours is not None else float(hcfg.get("minrepo_minutes", 20)) / 60
        n = mr.backfill_history(today_jst(), before=first, since=since, coef=coef,
                                deadline=time.time() + hours * 3600)
        log.info("みんレポ過去分 %d 台日を保存", n)
    elif a.cmd == "compare":
        r = cmd_compare(cfg, db, root, force=True)
        log.info("比較結果: %s", json.dumps((r or {}).get("summary", {}), ensure_ascii=False)[:2000])
    elif a.cmd == "page":
        # 取得はせず、いまのデータと最新のプログラムでページだけ作り直す（設定やプログラムを変えたとき用）
        run_checks(cfg, db, root)
        cmd_predict(cfg, db, root, today_jst())
    elif a.cmd == "select":
        r = cmd_select(cfg, db, root, force=True)
        log.info("選択結果: %s", json.dumps({k: v for k, v in (r or {}).items() if k != "single"}, ensure_ascii=False))
    elif a.cmd == "status":
        print(json.dumps(db.coverage(), ensure_ascii=False, indent=1))
        print("アナスロ未取得:", len(set(db.index("anaslo")) - db.fetched_dates("anaslo")), "日")
    elif a.cmd == "import-seed":
        cmd_import_seed(db, root)
    elif a.cmd == "diag":
        from . import diag
        r = diag.run(root)
        log.info("診断: %s", json.dumps(r, ensure_ascii=False)[:1500])
    if csv_mode and a.cmd in ("update", "backfill", "daily", "minrepo-history", "import-seed", "inbox"):
        db.export_csv(root / "data", keyword=cfg["target"].get("keyword", "ハナハナ"))
        log.info("データを保存しました")
    return 0


if __name__ == "__main__":
    sys.exit(main())
