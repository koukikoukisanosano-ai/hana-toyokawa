"""予想方法の比較と、一番よい方法での予想。

狙い: その日の全台（1000G以上回った台）を差枚で並べた順位に、予想の順位を一番近づける。
採点: 各日を「その日より前のデータだけ」で学習した予想で並べ、実際の差枚順位との順位相関（スピアマン）を取り、
      検証期間の平均で比べる。学習は7日ごとにやり直す（実際の運用と同じ）。

比べる方法
  ・benchmark_zero : 何もしない（全台同点）＝基準 0
  ・prev_rank      : 前日の差枚順位そのまま（「前日勝った台がまた勝つ」）
  ・prev_rank_rev  : 前日の差枚順位の逆（「前日負けた台が勝つ」）
  ・unit_mean      : その台のこれまでの平均順位（「よく勝つ台」）
  ・ridge          : 下の特徴すべての重み付き足し算（リッジ回帰）
  ・gbm            : 勾配ブースティング木（特徴の組み合わせも学習。過学習しにくい控えめな設定）
  ・ensemble       : ridge と gbm の平均

特徴（すべて予想する日より前の情報だけ）
  前日〜3営業日前の差枚・G数・差枚順位・REG確率、直近7/30日の平均（差枚・順位・G数・推定設定・プラス率）、
  連続凹み日数、前回プラスから何営業日、台のこれまでの平均順位、台番・台番末尾・島の端からの位置・機種、
  曜日・日付末尾・旧イベ日・ゾロ目・月初/月末・休業明け、前日の店全体（ハナハナ全台）の平均差枚
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import model as M

log = logging.getLogger("hana.ml")
METHODS = ["prev_rank", "prev_rank_rev", "unit_mean", "factor", "ridge", "gbm", "ensemble"]
# 予想する中身（どれが一番当たるかを比べる）: 列名 → (表示名, 2値か)
TARGETS = {
    "t_rank": ("差枚の順位", False),
    "t_diff": ("差枚の枚数", False),
    "t_plus": ("プラスで終わるか", True),
    "t_payout": ("出率（差枚÷投入枚数）", False),
    "t_eset": ("推定設定（高設定っぽさ）", False),
}
T_METHODS = ["prev", "unit_mean", "ridge", "gbm", "ensemble"]
LABEL = {
    "prev": "前日の値そのまま", 
    "benchmark_zero": "何もしない（基準）", "prev_rank": "前日の差枚順位そのまま", "prev_rank_rev": "前日の差枚順位の逆",
    "unit_mean": "台ごとのこれまでの平均順位", "factor": "店の癖の掛け算（これまでの方法）",
    "ridge": "全特徴の重み付き足し算（リッジ回帰）", "gbm": "勾配ブースティング木（機械学習）", "ensemble": "リッジと機械学習の平均",
}


# ---------------------------------------------------------------- 特徴づくり
def _streak(flags: np.ndarray) -> np.ndarray:
    out = np.zeros(len(flags))
    c = 0
    for i, f in enumerate(flags):
        out[i] = c
        c = c + 1 if f else 0
    return out


def _since(flags: np.ndarray, cap: int = 30) -> np.ndarray:
    out = np.full(len(flags), float(cap))
    last = None
    for i, f in enumerate(flags):
        out[i] = cap if last is None else min(i - last, cap)
        if f:
            last = i
    return out


def build_frame(units: pd.DataFrame, cfg: M.Settings, target: pd.Timestamp | None = None) -> pd.DataFrame:
    """台×営業日の表に特徴と答え（y=その日の差枚順位 0〜1）を付ける。target を渡すとその日の行（答えなし）も足す。"""
    df, _ = M.prepare(units, cfg)
    df = df[df["games"].notna()].copy()
    df["date"] = pd.to_datetime(df["date"])
    ok = df["games"] >= cfg.label_min_games
    r = df["diff"].where(ok).groupby(df["date"]).rank(method="average")
    n = ok.groupby(df["date"]).transform("sum")
    df["y"] = ((r - 0.5) / n).where(ok)
    g2 = df["games"] >= cfg.judge_min_games
    df["t_rank"] = df["y"]
    df["t_diff"] = df["diff"].where(ok)
    df["t_plus"] = (df["diff"] > 0).astype(float).where(ok)
    df["t_payout"] = (df["diff"] / (3 * df["games"].clip(lower=1))).where(g2)
    df["t_eset"] = df["e_set"].where(g2)
    df["rate_reg"] = df["rb"] / df["games"].clip(lower=1) * 1000
    df["plus"] = (df["diff"] > 0).astype(float)
    hall = df[ok].groupby("date")["diff"].mean().rename("hall_diff")

    days = sorted(df["date"].unique())
    if target is not None:
        t = pd.Timestamp(target)
        if t not in set(days):
            last = df[df["date"] == days[-1]][["unit", "model"]].copy()
            recent = df[df["date"] >= days[-1] - pd.Timedelta(days=7)].groupby("unit").tail(1)[["unit", "model"]]
            add = recent.assign(date=t)
            df = pd.concat([df, add], ignore_index=True)
            days.append(t)
    di = {d: i for i, d in enumerate(sorted(days))}
    df["di"] = df["date"].map(di)
    df = df.sort_values(["unit", "di"]).reset_index(drop=True)

    g = df.groupby("unit")
    for k in (1, 2, 3):
        same = (df["di"] - g["di"].shift(k)) == k
        for c in ("diff", "games", "y", "rate_reg"):
            df[f"{c}_l{k}"] = g[c].shift(k).where(same)
    for w in (7, 30):
        for c in ("diff", "y", "games", "e_set", "plus"):
            df[f"{c}_m{w}"] = g[c].transform(lambda s: s.shift(1).rolling(w, min_periods=max(3, w // 4)).mean())
    df["y_all"] = g["y"].transform(lambda s: s.shift(1).expanding(min_periods=10).mean())
    same1 = (df["di"] - g["di"].shift(1)) == 1
    for t in TARGETS:
        df[t + "_l1"] = g[t].shift(1).where(same1)
        df[t + "_all"] = g[t].transform(lambda s: s.shift(1).expanding(min_periods=10).mean())
    df["streak_low"] = np.concatenate([_streak((gg["diff"] <= -500).to_numpy()) for _, gg in g])
    df["since_plus"] = np.concatenate([_since((gg["diff"] > 0).to_numpy()) for _, gg in g])

    # 台の位置
    us = sorted(df["unit"].unique())
    runs, cur = [], []
    for u in us:
        if cur and u != cur[-1] + 1:
            runs.append(cur)
            cur = []
        cur.append(u)
    runs.append(cur)
    pos = {u: min(i, len(rn) - 1 - i) for rn in runs for i, u in enumerate(rn)}
    df["edge"] = df["unit"].map(pos).clip(upper=5)
    df["udigit"] = df["unit"] % 10
    df["model_c"] = df["model"].astype("category").cat.codes

    # 日の特徴
    dd = df["date"].dt
    df["wday"] = dd.weekday
    df["ddigit"] = dd.day % 10
    df["event"] = df["ddigit"].isin(cfg.event_digits).astype(int)
    df["zoro"] = dd.day.isin([11, 22]).astype(int)
    df["mstart"] = (dd.day <= 3).astype(int)
    df["mend"] = (dd.day >= 28).astype(int)
    gapd = df["date"].map(dict(zip(sorted(days), np.r_[1, np.diff(np.array(sorted(days), dtype="datetime64[D]")).astype(int)])))
    df["after_closed"] = (gapd > 1).astype(int)
    from .calendar_jp import is_busy, is_holiday
    dmap = {d: (int(is_busy(pd.Timestamp(d).date())), int(is_holiday(pd.Timestamp(d).date()))) for d in df["date"].unique()}
    df["busy"] = df["date"].map(lambda d: dmap[d][0])
    df["holiday"] = df["date"].map(lambda d: dmap[d][1])
    hall_prev = hall.shift(1)
    df["hall_prev"] = df["date"].map(hall_prev)
    return df.sort_values(["di", "unit"]).reset_index(drop=True)


FEATURES = (["diff_l1", "diff_l2", "diff_l3", "games_l1", "games_l2", "y_l1", "y_l2", "y_l3", "rate_reg_l1",
             "diff_m7", "diff_m30", "y_m7", "y_m30", "games_m7", "games_m30", "e_set_m7", "e_set_m30",
             "plus_m7", "plus_m30", "y_all", "streak_low", "since_plus",
             "unit", "udigit", "edge", "model_c",
             "wday", "ddigit", "event", "zoro", "mstart", "mend", "after_closed", "hall_prev", "busy", "holiday"])


# ---------------------------------------------------------------- 方法ごとの予想
def _fit_predict(method: str, tr: pd.DataFrame, te: pd.DataFrame, target: str | None = None) -> np.ndarray:
    if target is not None:      # 指定の中身（t_*）を答えにして学習
        if method in ("ridge", "gbm"):
            return _fit_model(method, tr, target)(te)
        if method == "prev":
            v = te[target + "_l1"]
            return v.fillna(v.mean() if v.notna().any() else 0).to_numpy()
        if method == "unit_mean":
            v = te[target + "_all"]
            return v.fillna(v.mean() if v.notna().any() else 0).to_numpy()
        tr = tr[tr[target].notna()].assign(y=lambda d: d[target])
        if method == "ensemble":
            a, b = _fit_predict("ridge", tr, te), _fit_predict("gbm", tr, te)
            return (pd.Series(a).rank().to_numpy() + pd.Series(b).rank().to_numpy()) / 2 / len(a)
        return _fit_predict(method, tr, te)
    if method == "prev_rank":
        return te["y_l1"].fillna(0.5).to_numpy()
    if method == "prev_rank_rev":
        return 1 - te["y_l1"].fillna(0.5).to_numpy()
    if method == "unit_mean":
        return te["y_all"].fillna(0.5).to_numpy()
    X, y = tr[FEATURES], tr["y"].to_numpy()
    if method == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        cats = ["wday", "ddigit", "udigit", "model_c"]
        Xt = pd.get_dummies(X.astype({c: "category" for c in cats}), columns=cats)
        Xe = pd.get_dummies(te[FEATURES].astype({c: "category" for c in cats}), columns=cats).reindex(columns=Xt.columns, fill_value=0)
        m = make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=300.0)).fit(Xt, y)
        return m.predict(Xe)
    if method == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.03, max_depth=3, min_samples_leaf=200,
                                          l2_regularization=1.0, random_state=0).fit(X, y)
        return m.predict(te[FEATURES])
    if method == "ensemble":
        a, b = _fit_predict("ridge", tr, te), _fit_predict("gbm", tr, te)
        return (pd.Series(a).rank().to_numpy() + pd.Series(b).rank().to_numpy()) / 2
    raise ValueError(method)


def _factor_scores(units, hall, cfg, test_days, refit_every=7) -> dict:
    """これまでの方法（店の癖の掛け算）の検証用スコア: 日付→(unit→score)"""
    from . import backtest as BT
    df, rows = BT._prep(units, hall, cfg, 1)
    X = M.build_X(rows, df, hall, "cut")
    M.neutralize_prev(X, rows)
    cols = list(M.FACTORS) + list(M.HALL_FACTORS)
    first = df["date"].min() + pd.Timedelta(days=21)
    out, beta, last = {}, None, None
    n_hall = len(M.HALL_FACTORS)
    for d in test_days:
        if beta is None or (d - last).days >= refit_every:
            m = (rows["date"] < d) & (rows["date"] >= first) & (rows["w"] > 0)
            tr = rows[m]
            base = float((tr["w"] * tr["y"]).sum() / max(tr["w"].sum(), 1e-9))
            beta = M.fit_combiner(X.loc[m, cols].to_numpy(), tr["y"].to_numpy(), tr["w"].to_numpy(), cfg.l2,
                                  float(M._logit(base)), n_hall) if m.sum() > 200 else None
            last = d
        if beta is None:
            continue
        day = rows["date"] == d
        A = np.hstack([np.ones((int(day.sum()), 1)), X.loc[day, cols].to_numpy()])
        out[d] = dict(zip(rows.loc[day, "unit"], A @ beta))
    return out


def compare(units: pd.DataFrame, hall: pd.DataFrame, cfg: M.Settings, test_days: int = 400,
            refit_every: int = 7, methods=None) -> dict:
    """各方法を、検証期間の各日について「その日より前のデータだけ」で学習して予想し、成績を比べる。"""
    methods = methods or METHODS
    fr = build_frame(units, cfg)
    days = sorted(fr.loc[fr["y"].notna(), "date"].unique())
    min_train = 60
    if len(days) < min_train + 20:
        return {"ok": False, "reason": f"データが{len(days)}日分です（{min_train + 20}日分で比較できます）"}
    tdays = [pd.Timestamp(d) for d in days[-test_days:] if d >= days[min_train]]
    fac = _factor_scores(units, hall, cfg, tdays, refit_every) if "factor" in methods else {}
    res = {m: [] for m in methods}
    preds: dict[str, dict] = {m: {} for m in methods}
    cache_day = None
    for i, d in enumerate(tdays):
        te = fr[(fr["date"] == d) & fr["y"].notna()]
        if len(te) < 10:
            continue
        if cache_day is None or (d - cache_day).days >= refit_every:
            tr = fr[(fr["date"] < d) & fr["y"].notna()]
            cache_day, cache_tr = d, tr
        for m in methods:
            if m == "factor":
                s = te["unit"].map(fac.get(d, {})).to_numpy(dtype=float)
                if np.isnan(s).all():
                    continue
                s = np.nan_to_num(s, nan=np.nanmean(s))
            else:
                s = _fit_predict(m, cache_tr, te)
            rho = pd.Series(s).rank().corr(pd.Series(te["y"].to_numpy()).rank()) if np.nanstd(s) > 0 else 0.0
            top = te.assign(s=s).sort_values("s", ascending=False).head(5)
            res[m].append({"date": d, "rho": 0.0 if pd.isna(rho) else float(rho),
                           "top5_diff": float(top["diff"].mean()), "all_diff": float(te["diff"].mean()),
                           "top5_plus": float((top["diff"] > 0).mean()), "all_plus": float((te["diff"] > 0).mean())})
    summary = {}
    for m, rows in res.items():
        if not rows:
            continue
        t = pd.DataFrame(rows)
        half = len(t) // 2
        summary[m] = {"label": LABEL[m], "rho": round(float(t["rho"].mean()), 4),
                      "se": round(float(t["rho"].std() / np.sqrt(len(t))), 4),
                      "rho_first": round(float(t["rho"][:half].mean()), 4), "rho_second": round(float(t["rho"][half:].mean()), 4),
                      "pos_days": round(float((t["rho"] > 0).mean()), 3),
                      "top5_diff": round(float(t["top5_diff"].mean()), 1), "all_diff": round(float(t["all_diff"].mean()), 1),
                      "top5_plus": round(float(t["top5_plus"].mean()), 3), "all_plus": round(float(t["all_plus"].mean()), 3),
                      "days": len(t)}
    # 一番よい方法: 平均の一致度が最大、かつ前半・後半の両方でプラス。どれも満たさなければ「何もしない」
    good = {m: v for m, v in summary.items() if v["rho_first"] > 0 and v["rho_second"] > 0}
    best = max(good, key=lambda m: good[m]["rho"]) if good else "benchmark_zero"
    return {"ok": True, "best": best, "summary": summary, "test_days": len(tdays), "n_days": len(days)}


def _concord(s: np.ndarray, y: np.ndarray, binary: bool) -> float:
    """2台を比べて、予想で上にした方が実際にも上だった割合（0.5＝コイン投げ）。2値のときはAUC。"""
    from scipy.stats import kendalltau
    if np.nanstd(s) == 0:
        return 0.5
    if binary:
        if y.min() == y.max():
            return np.nan
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, s))
    tau = kendalltau(s, y).correlation
    return 0.5 if np.isnan(tau) else (tau + 1) / 2


def _fit_model(method: str, tr: pd.DataFrame, target: str):
    """ridge / gbm を学習して、予想する関数を返す（検証では14日ごとに1回だけ学習する）。"""
    tr = tr[tr[target].notna()]
    X, y = tr[FEATURES], tr[target].to_numpy()
    if method == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.03, max_depth=3, min_samples_leaf=200,
                                          l2_regularization=1.0, random_state=0).fit(X, y)
        return lambda te: m.predict(te[FEATURES])
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cats = ["wday", "ddigit", "udigit", "model_c"]
    Xt = pd.get_dummies(X.astype({c: "category" for c in cats}), columns=cats)
    cols = Xt.columns
    m = make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=300.0)).fit(Xt, y)
    return lambda te: m.predict(pd.get_dummies(te[FEATURES].astype({c: "category" for c in cats}), columns=cats)
                                .reindex(columns=cols, fill_value=0))


def compare_all(units: pd.DataFrame, cfg: M.Settings, test_days: int = 400, refit_every: int = 14) -> dict:
    """予想する中身5種類 × 方法5通りを、同じ物差し（2台比べて当たる割合）で比べる。
    採用: 当たる割合の平均がコイン投げより上で、それが偶然でない度合い（t値）が一番大きいもの。
          ただし検証期間の前半・後半の両方で 0.5 を上回ったものに限る。"""
    fr = build_frame(units, cfg)
    days = sorted(fr.loc[fr["t_rank"].notna(), "date"].unique())
    if len(days) < 80:
        return {"ok": False, "reason": f"データが{len(days)}日分です（80日分で比較できます）"}
    tdays = [pd.Timestamp(d) for d in days[-test_days:] if d >= days[60]]
    res = {(t, m): [] for t in TARGETS for m in T_METHODS}
    cache_day, fitted = None, {}
    for d in tdays:
        if cache_day is None or (d - cache_day).days >= refit_every:
            cache_day = d
            tr = fr[fr["date"] < d]
            fitted = {(t, m): _fit_model(m, tr, t) for t in TARGETS for m in ("ridge", "gbm")}
        day = fr[fr["date"] == d]
        for t, (lab, binary) in TARGETS.items():
            te = day[day[t].notna()]
            if len(te) < 10:
                continue
            yv = te[t].to_numpy()
            pr = {"ridge": fitted[(t, "ridge")](te), "gbm": fitted[(t, "gbm")](te)}
            pr["ensemble"] = (pd.Series(pr["ridge"]).rank().to_numpy() + pd.Series(pr["gbm"]).rank().to_numpy()) / 2 / len(te)
            for m in ("prev", "unit_mean"):
                pr[m] = _fit_predict(m, None, te, target=t)
            for m in T_METHODS:
                c = _concord(np.nan_to_num(pr[m]), yv, binary)
                if not np.isnan(c):
                    res[(t, m)].append(c - 0.5)
    summary = {}
    for (t, m), v in res.items():
        if len(v) < 20:
            continue
        v = np.array(v)
        h = len(v) // 2
        se = float(v.std() / np.sqrt(len(v))) or 1e-9
        summary[f"{t}|{m}"] = {"target": t, "method": m, "target_label": TARGETS[t][0], "label": "台ごとのこれまでの平均" if m == "unit_mean" else LABEL[m],
                               "acc": round(0.5 + float(v.mean()), 4), "t": round(float(v.mean()) / se, 2),
                               "first": round(0.5 + float(v[:h].mean()), 4), "second": round(0.5 + float(v[h:].mean()), 4),
                               "days": len(v)}
    good = {k: v for k, v in summary.items() if v["first"] > 0.5 and v["second"] > 0.5}
    best = max(good, key=lambda k: good[k]["t"]) if good else None
    return {"ok": True, "best": best, "summary": summary, "test_days": len(tdays), "n_days": len(days)}


def predict_target(units: pd.DataFrame, cfg: M.Settings, target: str, method: str, day) -> pd.DataFrame | None:
    """選ばれた中身・方法で、対象日の全台の予想値を出す。"""
    fr = build_frame(units, cfg, target=pd.Timestamp(day))
    t = pd.Timestamp(day)
    tr, te = fr[fr["date"] < t], fr[fr["date"] == t]
    if te.empty or tr[target].notna().sum() < 1000:
        return None
    return pd.DataFrame({"unit": te["unit"].astype(int).to_numpy(), "ml_score": _fit_predict(method, tr, te, target=target)})


def predict(units: pd.DataFrame, cfg: M.Settings, method: str, target) -> pd.DataFrame | None:
    """一番よい方法で、対象日の全台の点数（大きいほど差枚が多くなりそう）を出す。"""
    if method in ("factor", "benchmark_zero"):
        return None
    fr = build_frame(units, cfg, target=pd.Timestamp(target))
    t = pd.Timestamp(target)
    tr = fr[(fr["date"] < t) & fr["y"].notna()]
    te = fr[fr["date"] == t]
    if te.empty or len(tr) < 1000:
        return None
    s = _fit_predict(method, tr, te)
    return pd.DataFrame({"unit": te["unit"].astype(int).to_numpy(), "ml_score": s})


def explain(units: pd.DataFrame, cfg: M.Settings, target) -> dict[int, list[str]]:
    """台ごとの、予想の材料になった主な数字（表示用）。"""
    fr = build_frame(units, cfg, target=pd.Timestamp(target))
    te = fr[fr["date"] == pd.Timestamp(target)]
    out = {}
    for _, r in te.iterrows():
        items = []
        if pd.notna(r["y_l1"]):
            items.append(f"前日 差枚順位 上位{(1 - r['y_l1']) * 100:.0f}%")
        if pd.notna(r["y_m7"]):
            items.append(f"直近7日 平均 上位{(1 - r['y_m7']) * 100:.0f}%")
        if pd.notna(r["games_m7"]):
            items.append(f"直近7日 平均{r['games_m7']:,.0f}G")
        if r["streak_low"] >= 2:
            items.append(f"凹み{int(r['streak_low'])}日連続")
        out[int(r["unit"])] = items
    return out
