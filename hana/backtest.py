"""過去の日で「その日より前のデータだけで予想していたら当たっていたか」を検証する。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import model as M


def _evaluate(rows, X, cols, cfg, test_dates, lag, top_k, refit_every, first) -> list[dict]:
    """各テスト日を、その日より前のデータだけで学習した重みで予想し、上位 top_k 台の成績を記録する。"""
    n_hall = sum(1 for c in cols if c.startswith("h_"))
    n_f = len(cols) - n_hall
    beta, last_fit = None, None
    per_day = []
    for d in test_dates:
        if beta is None or (d - last_fit).days >= refit_every:
            train_m = (rows["date"] <= d - pd.Timedelta(days=lag)) & (rows["date"] >= first) & (rows["w"] > 0)
            tr = rows[train_m]
            base = float((tr["w"] * tr["y"]).sum() / max(tr["w"].sum(), 1e-9))
            if train_m.sum() > 200 and cols:
                beta = M.fit_combiner(X.loc[train_m, cols].to_numpy(), tr["y"].to_numpy(), tr["w"].to_numpy(),
                                      cfg.l2, float(M._logit(base)), n_hall)
            else:
                beta = np.r_[M._logit(base), np.ones(n_f), np.zeros(n_hall)]
            last_fit = d
        day = rows["date"] == d
        dd = rows[day & (rows["games"] >= cfg.label_min_games)].copy()
        if len(dd) < top_k * 2:
            continue
        A = np.hstack([np.ones((len(dd), 1)), X.loc[dd.index, cols].to_numpy()]) if cols else np.ones((len(dd), 1))
        dd["score"] = 1 / (1 + np.exp(-(A @ beta)))
        # 同点（要因なし等）のときに台番順で有利不利が出ないよう、同点はランダムに並べる
        dd["_r"] = np.random.default_rng(int(pd.Timestamp(d).value % 2**32)).random(len(dd))
        dd = dd.sort_values(["score", "_r"], ascending=False)
        top = dd.head(top_k)
        best = set(dd.sort_values("e_set", ascending=False).head(top_k)["unit"])
        per_day.append({
            "date": d.date(),
            "n": len(dd),
            "top_p": top["p_high"].mean(), "all_p": dd["p_high"].mean(),
            "top_e": top["e_set"].mean(), "all_e": dd["e_set"].mean(),
            "top_diff": top["diff"].mean(), "all_diff": dd["diff"].mean(),
            "hit": len(best & set(top["unit"])) / top_k,
            "rand_hit": top_k / len(dd),
            "picks": ",".join(str(u) for u in top["unit"]),
        })
    return per_day


def _prep(units, hall, cfg, lag):
    df, prior = M.prepare(units, cfg)
    df = M.add_prev(df, cfg)
    pos = M.corners(df.loc[df["date"] == df["date"].max(), "unit"].tolist(), cfg.islands)
    df = M.add_keys(df, cfg, pos)
    rows = df.copy()
    rows["cut"] = rows["date"] - pd.Timedelta(days=lag)
    if lag > 1:  # 前日データが無い想定
        rows.loc[:, "prev_p"] = np.nan
    return df, rows


def select_factors(units: pd.DataFrame, hall: pd.DataFrame, cfg: M.Settings, days: int = 90, lag: int = 1,
                   top_k: int = 5, refit_every: int = 7, min_gain: float = 0.004, max_factors: int = 8) -> dict:
    """この店のデータで、どの要因を使うと予想が良くなるかを検証して選ぶ（前向き選択）。

    評価: 各日を「その日より前のデータだけ」で予想し、上位 top_k 台の高設定確率の平均（全台平均との差）。
    何も使わない状態から、一番成績が上がる要因を1つずつ足していき、上がり幅が min_gain 未満になったら止める。
    """
    df, rows = _prep(units, hall, cfg, lag)
    all_dates = sorted(df["date"].unique())
    need = cfg.min_train_days + 15
    if len(all_dates) < need:
        return {"ok": False, "reason": f"データが{len(all_dates)}日分です（{need}日分たまったら、この店に合う要因を自動で選びます）"}
    names = list(M.ALL_FACTORS) + list(M.ALL_HALL_FACTORS)
    X = M.build_X(rows, df, hall, "cut", names=names)
    # 台の順位を決める要因だけを選ぶ（日の要因は順位を変えないので常に使う）
    cands = [c for c in names if c not in M.DAY_FACTORS]
    M.neutralize_prev(X, rows)
    first = df["date"].min() + pd.Timedelta(days=21)
    test_dates = [d for d in all_dates[-days:] if d >= df["date"].min() + pd.Timedelta(days=cfg.min_train_days)]

    def order(cs):
        return [c for c in names if c in cs]

    def score(cs):
        """(全体, 奇数番目の日, 偶数番目の日) の成績。偶然を拾わないよう、両方で良くなることを条件にする。"""
        pd_ = _evaluate(rows, X, order(cs), cfg, test_dates, lag, top_k, refit_every, first)
        if not pd_:
            return (float("nan"),) * 3
        v = (pd.DataFrame(pd_)["top_p"] - pd.DataFrame(pd_)["all_p"]).to_numpy()
        return float(v.mean()), float(v[0::2].mean()), float(v[1::2].mean())

    chosen: list[str] = []
    cur = score(chosen)
    base_score = cur[0]
    single = {c: score([c])[0] - base_score for c in cands}
    steps = []
    while len(chosen) < max_factors:
        best_c, best_s = None, cur
        for c in cands:
            if c in chosen:
                continue
            sc = score(chosen + [c])
            if sc[0] > best_s[0] and sc[1] > cur[1] and sc[2] > cur[2]:
                best_c, best_s = c, sc
        if best_c is None or best_s[0] - cur[0] < min_gain:
            break
        chosen.append(best_c)
        steps.append({"add": best_c, "label": M.factor_label(best_c), "gain": round(best_s[0] - cur[0], 4)})
        cur = best_s
    cur = cur[0]
    return {"ok": True, "active": order(chosen + M.DAY_FACTORS), "unit_factors": order(chosen), "score": round(cur, 4), "base_score": round(base_score, 4),
            "steps": steps, "single": {k: round(v, 4) for k, v in single.items()},
            "test_days": len(test_dates), "n_days": len(all_dates), "top_k": top_k}


def run(units: pd.DataFrame, hall: pd.DataFrame, cfg: M.Settings, days: int = 60, lag: int = 1,
        top_k: int = 3, refit_every: int = 7) -> dict:
    df, rows = _prep(units, hall, cfg, lag)
    all_dates = sorted(df["date"].unique())
    if len(all_dates) < cfg.min_train_days + 5:
        return {"ok": False, "reason": f"データが{len(all_dates)}日分しかありません（{cfg.min_train_days + 5}日以上で検証できます）"}

    X = M.build_X(rows, df, hall, "cut")
    M.neutralize_prev(X, rows)
    cols = list(M.FACTORS) + list(M.HALL_FACTORS)
    first = df["date"].min() + pd.Timedelta(days=21)

    test_dates = [d for d in all_dates[-days:] if d >= df["date"].min() + pd.Timedelta(days=cfg.min_train_days)]
    per_day = _evaluate(rows, X, cols, cfg, test_dates, lag, top_k, refit_every, first)
    if not per_day:
        return {"ok": False, "reason": "検証できる日がありませんでした"}
    t = pd.DataFrame(per_day)
    # 順位相関（予想スコア vs 推定設定）の平均
    return {
        "ok": True, "days": len(t), "lag": lag, "top_k": top_k,
        "top_e": t["top_e"].mean(), "all_e": t["all_e"].mean(),
        "top_p": t["top_p"].mean(), "all_p": t["all_p"].mean(),
        "top_diff": t["top_diff"].mean(), "all_diff": t["all_diff"].mean(),
        "hit": t["hit"].mean(), "rand_hit": t["rand_hit"].mean(),
        "win_days": float((t["top_diff"] > t["all_diff"]).mean()),
        "table": t,
    }
