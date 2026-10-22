"""店全体の「日の強さ」: 日の種類ごとの、ハナハナ全台の出率（ボーナス回数から計算）。

この店で1年4か月分のデータから見つかった傾向（2025年の傾向で2026年も当たることを確認済み）:
  ・混む日（土日・祝日・GW・お盆・年末年始）は出率が低い（平日より約0.6%低い）
  ・ただし混む日でも 0・5 のつく日（旧イベ日）は下がらない
  ・平日の中では火曜がやや高い
日の種類ごとの平均を、日数が少ないときは全体平均に寄せて見積もる（毎回データから計算し直す）。
"""
from __future__ import annotations

from datetime import date as Date, timedelta

import numpy as np
import pandas as pd

from .calendar_jp import is_busy, is_holiday, is_long_holiday

WEEKDAY = "月火水木金土日"


def category(d: Date, event_digits=(0, 5)) -> str:
    ev = d.day % 10 in event_digits
    if is_busy(d):
        return "混む日・旧イベ日" if ev else "混む日"
    return "平日・火曜" if d.weekday() == 1 else "平日"


def why(d: Date) -> str:
    if is_long_holiday(d):
        return "連休"
    if is_holiday(d):
        return "祝日"
    if d.weekday() >= 5:
        return f"{WEEKDAY[d.weekday()]}曜"
    return f"{WEEKDAY[d.weekday()]}曜"


def estimate(hist: pd.DataFrame, target: Date, event_digits=(0, 5), days: int = 7, m: float = 15.0) -> dict:
    """日の種類ごとの出率（その種類の日の全台の 差枚合計÷投入合計）。日数が少ない種類は全体に寄せる。"""
    h = hist[hist["games"] > 0].dropna(subset=["diff"])
    if h.empty:
        return {}
    t = h.assign(inn=3 * h["games"]).groupby("date").agg(diff=("diff", "sum"), inn=("inn", "sum")).reset_index()
    t["date"] = pd.to_datetime(t["date"])
    if len(t) < 30:
        return {}
    t["cat"] = [category(d.date(), event_digits) for d in t["date"]]
    base = float(t["diff"].sum() / t["inn"].sum() * 100)
    g = t.groupby("cat").agg(diff=("diff", "sum"), inn=("inn", "sum"), n=("diff", "size"))
    g["pay"] = g["diff"] / g["inn"] * 100
    g["est"] = (g["pay"] * g["n"] + base * m) / (g["n"] + m)

    def one(d: Date) -> dict:
        c = category(d, event_digits)
        est = float(g["est"].get(c, base))
        return {"date": d, "cat": c, "why": why(d), "pay": 100 + est, "delta": est - base,
                "n": int(g["n"].get(c, 0)), "exp8000": est / 100 * 24000}

    table = [{"cat": c, "pay": 100 + float(r["est"]), "n": int(r["n"]), "exp8000": float(r["est"]) / 100 * 24000}
             for c, r in g.sort_values("est", ascending=False).iterrows()]
    return {"base": 100 + base, "n_days": int(len(t)), "today": one(target),
            "ahead": [one(target + timedelta(days=k)) for k in range(days)], "table": table}


def level(delta: float) -> tuple[str, str]:
    if delta >= 0.35:
        return "強い", "hot"
    if delta >= 0.12:
        return "やや強い", "warm"
    if delta > -0.15:
        return "普通", "norm"
    return "弱い", "cold"
