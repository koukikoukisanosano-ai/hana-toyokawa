"""続けるか・移るか・やめるか（撤退の判断）。

考え方
  ハナハナは毎ゲームが独立した抽選で天井もない。打ち続けるかどうかは「この台の設定」の見込みだけで決まる。
  途中の G数・BIG回数・REG回数 から、設定ごとの「その数字になる確率」を計算し、
  打つ前の見込み（予想順位ごとの過去の実績と、この店の設定配分）と組み合わせて「この先の見込み出率」を出す。
    ・この先の見込み出率が100%未満          → やめる（打つほど負けの見込み）
    ・空いている別の台の見込みの方が高い     → 移る
    ・それ以外                              → 続ける
  やめ方（負けたら早くやめる人が多い）の偏りは、この計算には影響しない。

設定配分は、直近1年の台ごと・日ごとの BIG・REG 回数から機種ごとに推定する（data/setting_prior.json、毎週やり直し）。
"""
from __future__ import annotations

import json
import math
from datetime import date as Date
from pathlib import Path

import numpy as np
import pandas as pd

from .specs import spec_for, reload_prior


def _loglik(g: pd.DataFrame, sp):
    S = [i for i, b in enumerate(sp.bb) if b]
    G = g["games"].to_numpy(float)[:, None]
    B = g["bb"].to_numpy(float)[:, None]
    R = g["rb"].to_numpy(float)[:, None]
    pb = np.array([1 / sp.bb[i] for i in S])[None, :]
    pr = np.array([1 / sp.rb[i] for i in S])[None, :]
    return B * np.log(G * pb) - G * pb + R * np.log(G * pr) - G * pr, S


def fit_setting_prior(units: pd.DataFrame, root: Path, keyword: str = "ハナハナ", days: int = 365) -> dict:
    """機種ごとの設定配分を、BIG・REG 回数から推定して保存する（EM法）。"""
    d = units[units["model"].str.contains(keyword, na=False)].dropna(subset=["games", "bb", "rb"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d[(d["date"] >= d["date"].max() - pd.Timedelta(days=days)) & (d["games"] >= 300)]
    out = {}
    for name, g in d.groupby(d["model"].map(lambda m: spec_for(m).name)):
        sp = spec_for(g["model"].iloc[0])
        if not sp.verified or len(g) < 300:
            continue
        L, S = _loglik(g, sp)
        w = np.full(len(S), 1 / len(S))
        for _ in range(800):
            m = L.max(1, keepdims=True)
            P = np.exp(L - m) * w
            P /= P.sum(1, keepdims=True)
            w_new = P.mean(0)
            if np.abs(w_new - w).max() < 1e-7:
                w = w_new
                break
            w = w_new
        full = [0.0] * len(sp.bb)
        for i, x in zip(S, w):
            full[i] = round(float(x), 4)
        out[name] = full
    res = {"when": str(Date.today()), "days": days, "weights": out}
    (root / "data" / "setting_prior.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    reload_prior()
    return res


def tilt(w: list[float], pay: list[float], target: float) -> list[float]:
    """設定配分の形を保ったまま、平均出率が target(%) になるように傾ける。"""
    w = np.asarray(w, float)
    pay = np.asarray(pay, float)
    ok = w > 0
    if not ok.any():
        return list(w)
    lo, hi = -5.0, 5.0
    for _ in range(80):
        lam = (lo + hi) / 2
        x = np.where(ok, w * np.exp(lam * (pay - 100)), 0)
        m = (x * pay).sum() / x.sum()
        if m < target:
            lo = lam
        else:
            hi = lam
    x = np.where(ok, w * np.exp(lam * (pay - 100)), 0)
    return [round(float(v), 6) for v in x / x.sum()]


def posterior_pay(sp, w, G, B, R) -> float:
    """途中の G数・BIG・REG から、この先の見込み出率(%)。"""
    best, logs = -1e300, []
    for x, b, r, p in zip(w, sp.bb, sp.rb, sp.payout):
        if not b or x <= 0:
            logs.append(None)
            continue
        lb, lr = G / b, G / r
        v = math.log(x) + (B * math.log(lb) - lb + R * math.log(lr) - lr if G > 0 else 0.0)
        logs.append(v)
        best = max(best, v)
    num = den = 0.0
    for v, p in zip(logs, sp.payout):
        if v is None:
            continue
        e = math.exp(v - best)
        num += e * p
        den += e
    return num / den


def page_params(units: pd.DataFrame, summary: dict, day_delta: float, keyword: str = "ハナハナ") -> dict:
    """予想ページの計算機に埋め込む数字。順位ごとの見込み（過去の実績＋今日の日の強さ）を機種ごとに用意する。"""
    d = units[units["model"].str.contains(keyword, na=False)]
    last = d["date"].max()
    lineup = sorted(d[d["date"] == last]["model"].unique())
    models = {}
    for m in lineup:
        sp = spec_for(m)
        from .specs import setting_weights
        base = setting_weights(sp)
        buckets = []
        for b in summary.get("buckets", []):
            target = 100 + b.get("pay_s", b["pay"]) * 100 + day_delta
            buckets.append({"from": b["from"], "to": b["to"], "mean": round(target, 3),
                            "w": tilt(base, [p if p else 0 for p in sp.payout], target)})
        models[m] = {"label": m.split("‐")[0].split("-")[0], "bb": [b or 0 for b in sp.bb], "rb": [r or 0 for r in sp.rb],
                     "pay": [p or 0 for p in sp.payout], "buckets": buckets}
    return {"models": models, "day_delta": day_delta}


MOVE_MARGIN = 0.4    # 移るのは、候補の台の見込みが0.4%以上高いとき（あと4000G打つなら約50枚の差。小さな差で動き回らない）


def quick_table(params: dict, checkpoints=(2000, 4000, 6000)) -> list[dict]:
    """早見表: 予想上位の台で、BIG・REGがどちらも平均の何%以下になったら「移る」「やめる」か。"""
    rows = []
    for m, P in params["models"].items():
        sp = spec_for(m)
        top, alt = P["buckets"][0], (P["buckets"][2] if len(P["buckets"]) > 2 else P["buckets"][-1])
        for G in checkpoints:
            pb = sum(w / b for w, b in zip(top["w"], P["bb"]) if b)
            pr = sum(w / r for w, r in zip(top["w"], P["rb"]) if r)
            eb, er = G * pb, G * pr
            line = {}
            for key, thr in (("stop", 100.0), ("move", alt["mean"] - MOVE_MARGIN)):
                hit = None
                for k in [x / 100 for x in range(100, -1, -1)]:
                    if posterior_pay(sp, top["w"], G, round(eb * k), round(er * k)) < thr:
                        hit = (round(eb * k), round(er * k), k)
                        break
                line[key] = hit
            rows.append({"model": P["label"], "G": G, "avg": (round(eb), round(er)), **line,
                         "top_mean": top["mean"], "alt_mean": alt["mean"]})
    return rows
