"""予想システム（台選び＋日選び）の作成・検証・予想。

■ 何を当てるか
  台選び: その日、どの台に座れば出率（＝払い出し÷投入）が高いか。
          出率はBIG・REG回数とG数から計算する（差枚＝BIG回数×BIG枚数＋REG回数×REG枚数−G数×1Gあたり消費）。
          ボーナス回数から計算した出率は、他の人がどれだけ回したか（やめ方）の影響を受けにくい。
  日選び: その日のハナハナ全体の出率（行くべき日かどうか）。

■ 採点（過去の各日を、その日より前のデータだけで予想し直す）
  台選び: 予想上位k台の出率 − その日の全台の出率（期間全体でまとめて計算）。
          8000G打った場合の期待差枚の差に直して表示する（出率1%＝240枚）。
  日選び: 予想した日の出率と実際の出率の相関、強いと予想した日と弱いと予想した日の実際の差。
  偶然の幅: 日を入れ替えて計算し直す（ブートストラップ）。

■ 候補の方法（台選び）
  rand        : 適当に選ぶ（基準）
  prev        : 前日の出率が高かった台
  model_dt    : 機種×日の種類（混む日／平日）ごとの、これまでの出率
  unit_h*     : 台ごとのこれまでの出率（直近ほど重く。半減期 7/30/120日・全期間）
  hier_h*     : 機種×日の種類 ＋ 台ごとの出率
  ridge       : 全部の材料の重み付き足し算（リッジ回帰、G数で重み付け）
  gbm         : 勾配ブースティング木（材料の組み合わせも学習）
  ens         : いくつかの方法の順位の平均
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .calendar_jp import is_busy, is_holiday, is_long_holiday

log = logging.getLogger("hana.sysmodel")
HALF_LIVES = {"h7": 7, "h30": 30, "h120": 120, "hall": None}
SUM_WINDOWS = (1, 3, 5, 7, 10, 14, 21, 30)
DT_NAMES = {"busy_ev": "混む日・旧イベ日", "busy": "混む日", "tue": "平日・火曜", "wk": "平日"}


# ---------------------------------------------------------------- 表づくり
def _daytype(d, event_digits) -> str:
    ev = d.day % 10 in event_digits
    if is_busy(d):
        return "busy_ev" if ev else "busy"
    return "tue" if d.weekday() == 1 else "wk"


def base_table(units: pd.DataFrame, keyword: str = "ハナハナ", event_digits=(0, 5), target=None) -> pd.DataFrame:
    """台×営業日の表。diff（差枚）は機種ごとのBIG・REG枚数と機械割から計算し直す。target の日は答えなしで足す。"""
    from .sources.minrepo import est_diff
    d = units[units["model"].str.contains(keyword, na=False)].dropna(subset=["games", "bb", "rb"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    d["diff"] = [est_diff(b, r, g, m) for b, r, g, m in zip(d["bb"], d["rb"], d["games"], d["model"])]
    d = d.dropna(subset=["diff"])
    d["inn"] = 3.0 * d["games"]
    d["obs"] = True
    if target is not None:
        t = pd.Timestamp(target)
        if t not in set(d["date"]):
            last = d["date"].max()
            cur = d[d["date"] >= last - pd.Timedelta(days=7)].sort_values("date").groupby("unit").tail(1)
            add = cur[["unit", "model"]].assign(date=t, obs=False)
            d = pd.concat([d, add], ignore_index=True)
    d = d[["date", "unit", "model", "games", "bb", "rb", "diff", "inn", "obs"]].copy()
    d["obs"] = d["obs"].fillna(False).astype(bool)
    days = sorted(d["date"].unique())
    d["di"] = d["date"].map({x: i for i, x in enumerate(days)})
    info = {x: (_daytype(pd.Timestamp(x).date(), event_digits), is_busy(pd.Timestamp(x).date()),
                is_holiday(pd.Timestamp(x).date()), is_long_holiday(pd.Timestamp(x).date())) for x in days}
    d["dt"] = d["date"].map(lambda x: info[x][0])
    d["busy"] = d["date"].map(lambda x: int(info[x][1]))
    d["holiday"] = d["date"].map(lambda x: int(info[x][2]))
    d["longhol"] = d["date"].map(lambda x: int(info[x][3]))
    d["wday"] = d["date"].dt.weekday
    d["ddigit"] = d["date"].dt.day % 10
    d["ev"] = d["ddigit"].isin(event_digits).astype(int)
    o = d[d["obs"]]
    dp = o.groupby("date").apply(lambda g: g["diff"].sum() / g["inn"].sum(), include_groups=False)
    d["day_pay"] = d["date"].map(dp)
    d["r"] = np.where(d["obs"], d["diff"] - d["day_pay"] * d["inn"], np.nan)   # その日の平均より何枚多く出たか
    return d.sort_values(["di", "unit"]).reset_index(drop=True)


def _decayed(df: pd.DataFrame, key: list[str], cols: list[str], h: float | None) -> np.ndarray:
    """key ごとに、その行より前の行の cols を、半減期 h 営業日で重み付けして足した値（行×列）。"""
    order = df.sort_values(key + ["di"]).index.to_numpy()
    kv = df.loc[order, key].astype(str).agg("|".join, axis=1).to_numpy()
    di = df.loc[order, "di"].to_numpy(float)
    X = df.loc[order, cols].fillna(0.0).to_numpy(float)
    S = np.zeros(len(cols))
    prev_k, prev_di = None, 0.0
    res = np.zeros((len(order), len(cols)))
    for i in range(len(order)):
        if kv[i] != prev_k:
            S = np.zeros(len(cols))
            prev_k = kv[i]
        else:
            if h is not None:
                S = S * 0.5 ** ((di[i] - prev_di) / h)
        res[i] = S
        S = S + X[i]
        prev_di = di[i]
    # 並べ替えた順番の結果を、元の行（index）に戻す
    return pd.DataFrame(res, index=order).reindex(df.index).to_numpy()


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """すべて『その日より前』の情報だけで作る材料。"""
    d = d.copy()
    for name, h in HALF_LIVES.items():
        v = _decayed(d, ["unit", "model"], ["r", "inn"], h)
        d[f"ur_{name}"], d[f"ui_{name}"] = v[:, 0], v[:, 1]
        v = _decayed(d, ["unit"], ["r", "inn"], h)          # 機種が変わっても同じ台番として
        d[f"pr_{name}"], d[f"pi_{name}"] = v[:, 0], v[:, 1]
    # 直近N営業日の差枚の合計（店が設定を上げ下げするときに見ていそうな数字）
    d = d.sort_values(["unit", "model", "di"])
    gm = d.groupby(["unit", "model"])
    for n in SUM_WINDOWS:
        d[f"sd{n}"] = gm["diff"].transform(lambda s: s.shift(1).rolling(n, min_periods=1).sum()).fillna(0.0)
        d[f"sr{n}"] = gm["r"].transform(lambda s: s.shift(1).rolling(n, min_periods=1).sum()).fillna(0.0)
    for h in (3, 7, 14):
        d[f"ed{h}"] = _decayed(d, ["unit", "model"], ["diff"], h)[:, 0]
    # 前日（前の営業日）の同じ台
    d = d.sort_values(["unit", "di"])
    g = d.groupby("unit")
    same = (d["di"] - g["di"].shift(1)) == 1
    d["prev_pay"] = (g["r"].shift(1) / g["inn"].shift(1).clip(lower=1)).where(same)
    d["prev_games"] = g["games"].shift(1).where(same)
    d["prev_diff"] = g["diff"].shift(1).where(same)
    low = (d["diff"] <= -500).fillna(False).to_numpy()
    st = np.zeros(len(d))
    units = d["unit"].to_numpy()
    for i in range(1, len(d)):
        if units[i] == units[i - 1] and same.iloc[i]:
            st[i] = st[i - 1] + 1 if low[i - 1] else 0
    d["streak_low"] = st
    d["g_m7"] = g["games"].transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
    d = d.sort_values(["di", "unit"])
    # 機種×日の種類、機種全体（その日より前の累計）
    for key, nm in ((["model", "busy"], "md"), (["model"], "mm")):
        agg = d[d["obs"]].groupby(key + ["di"])[["r", "inn"]].sum().reset_index().sort_values("di")
        agg[["cr", "ci"]] = agg.groupby(key)[["r", "inn"]].cumsum()      # その日を含む累計
        q = d[key + ["di"]].reset_index()
        # その日より前（その日を含まない）で一番新しい累計を使う
        q = pd.merge_asof(q.sort_values("di"), agg[key + ["di", "cr", "ci"]].sort_values("di"),
                          on="di", by=key, direction="backward", allow_exact_matches=False)
        q = q.set_index("index").reindex(d.index)
        d[f"{nm}_r"], d[f"{nm}_i"] = q["cr"].fillna(0).to_numpy(), q["ci"].fillna(0).to_numpy()
    first = d.groupby("model")["di"].transform("min")
    d["model_age"] = (d["di"] - first).clip(upper=400)
    # 台の位置（その日の並びの端からの距離）
    pos = {}
    for di_, gg in d.groupby("di"):
        us = sorted(gg["unit"].unique())
        run = []
        for u in us + [None]:
            if u is not None and run and u == run[-1] + 1:
                run.append(u)
                continue
            for i, x in enumerate(run):
                pos[(di_, x)] = min(i, len(run) - 1 - i)
            run = [u] if u is not None else []
    d["edge"] = [min(pos.get((a, b), 5), 5) for a, b in zip(d["di"], d["unit"])]
    d["udigit"] = d["unit"] % 10
    return d


# ---------------------------------------------------------------- 台選びの方法
NUM_FEATS = (["prev_pay", "prev_games", "prev_diff", "streak_low", "g_m7", "model_age", "edge", "udigit", "busy", "ev",
              "holiday", "longhol", "wday"]
             + [f"ueb_{n}" for n in HALF_LIVES] + [f"ui_{n}" for n in HALF_LIVES]
             + [f"peb_{n}" for n in HALF_LIVES] + ["md_eb", "mm_eb"]
             + [f"sd{n}" for n in SUM_WINDOWS] + [f"sr{n}" for n in SUM_WINDOWS] + ["ed3", "ed7", "ed14"])


def add_eb(d: pd.DataFrame, k_unit: float = 3e5, k_model: float = 1e6) -> pd.DataFrame:
    """縮小推定（データが少ないほど0＝平均に寄せる）した出率の差。"""
    d = d.copy()
    for n in HALF_LIVES:
        d[f"ueb_{n}"] = d[f"ur_{n}"] / (d[f"ui_{n}"] + k_unit)
        d[f"peb_{n}"] = d[f"pr_{n}"] / (d[f"pi_{n}"] + k_unit)
    d["md_eb"] = d["md_r"] / (d["md_i"] + k_model)
    d["mm_eb"] = d["mm_r"] / (d["mm_i"] + k_model)
    return d


def _design(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    X = df[NUM_FEATS].astype(float).copy()
    for m in models:
        X[f"m_{m}"] = (df["model"] == m).astype(float)
        X[f"mb_{m}"] = ((df["model"] == m) & (df["busy"] == 1)).astype(float)
    return X


class Learned:
    """ridge / gbm（14日ごとに学習し直す）。目的: その日の平均との出率の差（G数で重み付け）。"""

    def __init__(self, kind: str):
        self.kind = kind

    def fit(self, tr: pd.DataFrame):
        tr = tr[tr["obs"] & (tr["inn"] > 0)]
        self.models = sorted(tr["model"].unique())
        X = _design(tr, self.models)
        y = (tr["r"] / tr["inn"]).to_numpy()
        w = tr["inn"].clip(upper=30000).to_numpy()
        if self.kind == "ridge":
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            self.m = make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=1000.0))
            self.m.fit(X, y, ridge__sample_weight=w)
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.m = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.03, max_depth=3,
                                                   min_samples_leaf=400, l2_regularization=5.0, random_state=0)
            self.m.fit(X, y, sample_weight=w)
        return self

    def score(self, te: pd.DataFrame) -> np.ndarray:
        return self.m.predict(_design(te, self.models))


SIMPLE = {
    "rand": ("適当に選ぶ（基準）", None),
    "prev": ("前日の出率が高かった台", lambda t: t["prev_pay"].fillna(0).to_numpy()),
    "model_dt": ("機種×日の種類のこれまでの出率", lambda t: t["md_eb"].to_numpy()),
    "model": ("機種ごとのこれまでの出率", lambda t: t["mm_eb"].to_numpy()),
}
for _n, _h in HALF_LIVES.items():
    _lab = "全期間" if _h is None else f"直近重視・半減期{_h}日"
    SIMPLE[f"unit_{_n}"] = (f"台ごとのこれまでの出率（{_lab}）", (lambda n: lambda t: t[f"ueb_{n}"].to_numpy())(_n))
    SIMPLE[f"pos_{_n}"] = (f"台番ごとのこれまでの出率・機種替えも通算（{_lab}）", (lambda n: lambda t: t[f"peb_{n}"].to_numpy())(_n))
    SIMPLE[f"hier_{_n}"] = (f"機種×日の種類＋台ごとの出率（{_lab}）",
                            (lambda n: lambda t: (t["md_eb"] + t[f"ueb_{n}"]).to_numpy())(_n))
    # 逆張り: 最近出ていない台ほど上（凹み台を上げる店の癖を狙う）
    SIMPLE[f"rev_{_n}"] = (f"最近出ていない台（逆張り・{_lab}）", (lambda n: lambda t: -t[f"ueb_{n}"].to_numpy())(_n))
    SIMPLE[f"revp_{_n}"] = (f"最近出ていない台番（逆張り・機種替えも通算・{_lab}）",
                            (lambda n: lambda t: -t[f"peb_{n}"].to_numpy())(_n))
    SIMPLE[f"mdrev_{_n}"] = (f"機種×日の種類＋最近出ていない台（{_lab}）",
                             (lambda n: lambda t: (t["md_eb"] - t[f"ueb_{n}"]).to_numpy())(_n))
SIMPLE["prev_rev"] = ("前日に出なかった台（逆張り）", lambda t: -t["prev_pay"].fillna(0).to_numpy())
for _n in SUM_WINDOWS:
    SIMPLE[f"csd{_n}"] = (f"直近{_n}日の差枚が少ない台（凹み台）", (lambda n: lambda t: -t[f"sd{n}"].to_numpy())(_n))
    SIMPLE[f"csr{_n}"] = (f"直近{_n}日、全台平均より出ていない台", (lambda n: lambda t: -t[f"sr{n}"].to_numpy())(_n))
for _h in (3, 7, 14):
    SIMPLE[f"ced{_h}"] = (f"差枚が少ない台（直近重視・半減期{_h}日）", (lambda h: lambda t: -t[f"ed{h}"].to_numpy())(_h))
SIMPLE["streak"] = ("凹みが続いている台", lambda t: t["streak_low"].to_numpy())
LEARNED = {"ridge": "全部の材料の重み付き足し算（リッジ回帰）", "gbm": "勾配ブースティング木（機械学習）"}
LABELS = {**{k: v[0] for k, v in SIMPLE.items()}, **LEARNED}


def _rank01(s: np.ndarray) -> np.ndarray:
    return pd.Series(s).rank(method="average").to_numpy() / max(len(s), 1)


def evaluate_units(d: pd.DataFrame, test_days, methods=None, refit_every: int = 14, ens=None,
                   ks=(1, 3, 5), seed: int = 0) -> pd.DataFrame:
    """各テスト日について、各方法の上位k台の成績（差枚・投入の合計）を記録する。返り値は 日×方法×k の行。"""
    methods = methods or (list(SIMPLE) + list(LEARNED))
    rng = np.random.default_rng(seed)
    out = []
    fitted, fit_day = {}, None
    for d0 in test_days:
        day = d[(d["date"] == d0) & d["obs"]]
        if len(day) < 10:
            continue
        if any(m in LEARNED for m in methods) and (fit_day is None or (d0 - fit_day).days >= refit_every):
            tr = d[(d["date"] < d0) & d["obs"]]
            fitted = {m: Learned(m).fit(tr) for m in methods if m in LEARNED}
            fit_day = d0
        tie = rng.random(len(day))
        scores = {}
        for m in methods:
            if m == "rand":
                scores[m] = np.zeros(len(day))
            elif m in SIMPLE:
                scores[m] = np.nan_to_num(SIMPLE[m][1](day))
            elif m in LEARNED:
                scores[m] = fitted[m].score(day)
        for name, parts in (ens or {}).items():
            scores[name] = np.mean([_rank01(scores[p]) for p in parts], axis=0)
        all_d, all_i = float(day["diff"].sum()), float(day["inn"].sum())
        for m, s in scores.items():
            order = np.lexsort((tie, -np.round(s, 12)))
            for k in ks:
                pick = day.iloc[order[:k]]
                out.append((d0, m, k, float(pick["diff"].sum()), float(pick["inn"].sum()), all_d, all_i,
                            float(pick["diff"].mean()), float(day["diff"].mean()), len(day)))
    return pd.DataFrame(out, columns=["date", "method", "k", "pd", "pi", "ad", "ai", "pdm", "adm", "n"])


def summarize_units(ev: pd.DataFrame, n_boot: int = 1000, seed: int = 1) -> pd.DataFrame:
    """方法×k ごとに: 出率の差（上位k台 − 全台、期間でまとめて計算）、その偶然の幅、8000G換算。"""
    rng = np.random.default_rng(seed)
    rows = []
    for (m, k), g in ev.groupby(["method", "k"]):
        g = g.sort_values("date")
        pd_, pi_, ad_, ai_ = (g[c].to_numpy() for c in ("pd", "pi", "ad", "ai"))
        lift = pd_.sum() / pi_.sum() - ad_.sum() / ai_.sum()
        idx = rng.integers(0, len(g), size=(n_boot, len(g)))
        bl = pd_[idx].sum(1) / pi_[idx].sum(1) - ad_[idx].sum(1) / ai_[idx].sum(1)
        h = len(g) // 2
        l1 = pd_[:h].sum() / pi_[:h].sum() - ad_[:h].sum() / ai_[:h].sum()
        l2 = pd_[h:].sum() / pi_[h:].sum() - ad_[h:].sum() / ai_[h:].sum()
        rows.append({"method": m, "k": k, "lift": lift, "se": float(bl.std()), "lo": float(np.quantile(bl, 0.025)),
                     "hi": float(np.quantile(bl, 0.975)), "first": l1, "second": l2,
                     "pick_pay": pd_.sum() / pi_.sum(), "all_pay": ad_.sum() / ai_.sum(),
                     "diff_gap": float((g["pdm"] - g["adm"]).mean()), "days": len(g),
                     "exp8000": lift * 24000})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 日選び
def day_frame(d: pd.DataFrame) -> pd.DataFrame:
    o = d.groupby(["date", "di"]).agg(diff=("diff", "sum"), inn=("inn", "sum"), dt=("dt", "first"), busy=("busy", "first"),
                                      holiday=("holiday", "first"), longhol=("longhol", "first"), wday=("wday", "first"),
                                      ddigit=("ddigit", "first"), ev=("ev", "first"), nobs=("obs", "sum")).reset_index()
    o["pay"] = np.where(o["nobs"] > 0, o["diff"] / o["inn"].replace(0, np.nan), np.nan)
    o = o.sort_values("di")
    o["pay_l1"] = o["pay"].shift(1)
    o["pay_m7"] = o["pay"].shift(1).rolling(7, min_periods=3).mean()
    o["pay_m30"] = o["pay"].shift(1).rolling(30, min_periods=10).mean()
    return o.reset_index(drop=True)


def _day_predict(method: str, tr: pd.DataFrame, te: pd.DataFrame, m: float = 15.0) -> np.ndarray:
    tr = tr.dropna(subset=["pay"])
    base = tr["pay"].mean()
    if method == "mean":
        return np.full(len(te), base)
    if method == "dt":       # 日の種類（混む日・旧イベ・火曜・平日）ごとの平均（日数が少なければ全体平均に寄せる）
        g = tr.groupby("dt")["pay"].agg(["sum", "count"])
        est = (g["sum"] + m * base) / (g["count"] + m)
        return te["dt"].map(est).fillna(base).to_numpy()
    if method == "dt_recent":   # 日の種類 ＋ 直近30日の店の調子
        a = _day_predict("dt", tr, te, m)
        lvl = tr.tail(60)["pay"].mean() - base
        return a + 0.5 * lvl
    if method == "ridge":
        from sklearn.linear_model import Ridge
        cols = ["busy", "holiday", "longhol", "ev"]
        def X(df):
            x = df[cols].astype(float).copy()
            for w in range(7):
                x[f"w{w}"] = (df["wday"] == w).astype(float)
            for k in range(10):
                x[f"d{k}"] = (df["ddigit"] == k).astype(float)
            x["l1"] = df["pay_l1"].fillna(base) - base
            x["m30"] = df["pay_m30"].fillna(base) - base
            return x
        r = Ridge(alpha=30.0).fit(X(tr), tr["pay"] - base)
        return base + r.predict(X(te))
    raise ValueError(method)


DAY_METHODS = {"mean": "毎日同じ（基準）", "dt": "日の種類（混む日・旧イベ日・火曜・平日）", "dt_recent": "日の種類＋直近の店の調子",
               "ridge": "曜日・日付末尾・祝日など全部の重み付き足し算"}


def evaluate_days(df: pd.DataFrame, test_days) -> pd.DataFrame:
    rows = []
    for d0 in test_days:
        tr = df[df["date"] < d0]
        te = df[df["date"] == d0]
        if te.empty or pd.isna(te["pay"].iloc[0]) or len(tr) < 60:
            continue
        rec = {"date": d0, "pay": float(te["pay"].iloc[0]), "diff": float(te["diff"].iloc[0]), "inn": float(te["inn"].iloc[0])}
        for m in DAY_METHODS:
            rec[m] = float(_day_predict(m, tr, te)[0])
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_days(ev: pd.DataFrame, n_boot: int = 1000, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for m in DAY_METHODS:
        x, y = ev[m].to_numpy(), ev["pay"].to_numpy()
        r = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 else 0.0
        hi = x >= np.median(x)
        def gap(ix):
            xx, dd, ii = x[ix], ev["diff"].to_numpy()[ix], ev["inn"].to_numpy()[ix]
            h = xx >= np.median(xx)
            if h.all() or (~h).all():
                return 0.0
            return dd[h].sum() / ii[h].sum() - dd[~h].sum() / ii[~h].sum()
        g = gap(np.arange(len(ev)))
        bs = [gap(rng.integers(0, len(ev), len(ev))) for _ in range(n_boot)]
        out.append({"method": m, "corr": r, "gap": g, "se": float(np.std(bs)), "days": len(ev),
                    "good_pay": ev["diff"][hi].sum() / ev["inn"][hi].sum() if hi.any() else np.nan,
                    "bad_pay": ev["diff"][~hi].sum() / ev["inn"][~hi].sum() if (~hi).any() else np.nan})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 本番用（予想・答え合わせ・店の癖）
MAIN_METHOD = "gbm"           # 2026/10/22 の比較で採用（その日より前のデータだけで2週間ごとに学習し直す）
RANK_BUCKETS = [(1, 1), (2, 3), (4, 5), (6, 10), (11, 20), (21, 40), (41, 99)]


def build(units: pd.DataFrame, keyword: str, event_digits, target=None) -> pd.DataFrame:
    return add_eb(add_features(base_table(units, keyword, event_digits, target)))


def explain_row(r) -> list[str]:
    """ランキングに出す、その台の主な材料。"""
    out = [str(r["model"]).split("‐")[0].split("-")[0]]
    for n, lab in ((7, "直近7日"), (30, "直近30日")):
        v = r.get(f"sd{n}")
        if pd.notna(v):
            out.append(f"{lab} {v:+,.0f}枚")
    if pd.notna(r.get("prev_diff")):
        out.append(f"前日 {r['prev_diff']:+,.0f}枚")
    return out


def predict_units(d: pd.DataFrame, target) -> pd.DataFrame | None:
    """対象日の全台を、学習した機械学習モデルで点数づけする（大きいほど出率が高そう）。"""
    t = pd.Timestamp(target)
    tr = d[(d["date"] < t) & d["obs"]]
    te = d[d["date"] == t]
    if te.empty or len(tr) < 5000:
        return None
    m = Learned(MAIN_METHOD).fit(tr)
    out = te.copy()
    out["sys_score"] = m.score(te)
    out["why"] = [explain_row(r) for _, r in out.iterrows()]
    return out[["unit", "model", "sys_score", "why", "sd7", "sd30", "prev_diff"]].reset_index(drop=True)


def backtest(d: pd.DataFrame, start="2025-03-01", refit_every: int = 14) -> pd.DataFrame:
    """過去の各日を、その日より前のデータだけで予想し直した結果（日ごと）。"""
    days = [pd.Timestamp(x) for x in sorted(d.loc[d["obs"], "date"].unique()) if x >= pd.Timestamp(start)]
    dfd = day_frame(d)
    rows, fit, fday = [], None, None
    for d0 in days:
        if fit is None or (d0 - fday).days >= refit_every:
            fit = Learned(MAIN_METHOD).fit(d[(d["date"] < d0) & d["obs"]])
            fday = d0
        day = d[(d["date"] == d0) & d["obs"]]
        if len(day) < 10:
            continue
        s = fit.score(day)
        order = np.argsort(-s, kind="stable")
        ranked = day.iloc[order]
        tr = dfd[dfd["date"] < d0]
        te = dfd[dfd["date"] == d0]
        rec = {"date": d0.date().isoformat(), "day_diff": float(day["diff"].sum()), "day_inn": float(day["inn"].sum()),
               "n": len(day), "dt": day["dt"].iloc[0],
               "day_pred": float(_day_predict("dt", tr, te)[0]) if len(tr) >= 60 else np.nan,
               "top1": int(ranked["unit"].iloc[0]), "top3": ",".join(str(int(u)) for u in ranked["unit"].iloc[:3])}
        for a, b in RANK_BUCKETS:
            seg = ranked.iloc[a - 1:b]
            rec[f"b{a}_d"], rec[f"b{a}_i"] = float(seg["diff"].sum()), float(seg["inn"].sum())
        for k in (1, 3, 5):
            seg = ranked.iloc[:k]
            rec[f"t{k}_d"], rec[f"t{k}_i"] = float(seg["diff"].sum()), float(seg["inn"].sum())
        rows.append(rec)
    h = pd.DataFrame(rows)
    if len(h):
        thr = h["day_pred"].expanding().median().shift(1)
        h["good"] = (h["day_pred"] >= thr.fillna(h["day_pred"].iloc[0])).astype(int)
    return h


def _pay(dsum, isum):
    return float(dsum) / float(isum) if isum else np.nan


def history_summary(h: pd.DataFrame) -> dict:
    """答え合わせの表: やり方ごとの出率、月ごとの成績、順位ごとの実績。"""
    if h.empty:
        return {}
    h = h.copy()
    h["date"] = pd.to_datetime(h["date"])
    end = h["date"].max()
    periods = [("全期間", h["date"].min()), ("直近半年", end - pd.Timedelta(days=182)), ("直近3か月", end - pd.Timedelta(days=91))]
    rng = np.random.default_rng(7)

    def boot_se(x, dcol, icol, n=500):
        a, b = x[dcol].to_numpy(), x[icol].to_numpy()
        if len(a) < 5:
            return np.nan
        idx = rng.integers(0, len(a), (n, len(a)))
        return float((a[idx].sum(1) / b[idx].sum(1)).std())

    strat = []
    for name, since in periods:
        x = h[h["date"] >= since]
        g = x[x["good"] == 1]
        rows = [("毎日行く・適当な台（全台平均）", _pay(x["day_diff"].sum(), x["day_inn"].sum()), len(x), np.nan),
                ("強い日だけ行く・適当な台", _pay(g["day_diff"].sum(), g["day_inn"].sum()), len(g), np.nan),
                ("毎日行く・予想1位の台", _pay(x["t1_d"].sum(), x["t1_i"].sum()), len(x), boot_se(x, "t1_d", "t1_i")),
                ("毎日行く・予想上位3台のどれか", _pay(x["t3_d"].sum(), x["t3_i"].sum()), len(x), boot_se(x, "t3_d", "t3_i")),
                ("強い日だけ行く・予想1位の台", _pay(g["t1_d"].sum(), g["t1_i"].sum()), len(g), boot_se(g, "t1_d", "t1_i")),
                ("強い日だけ行く・予想上位3台のどれか", _pay(g["t3_d"].sum(), g["t3_i"].sum()), len(g), boot_se(g, "t3_d", "t3_i"))]
        strat.append({"period": name, "since": since.date().isoformat(), "days": len(x),
                      "rows": [{"label": a, "pay": b, "days": c, "se": e} for a, b, c, e in rows]})
    h["ym"] = h["date"].dt.strftime("%Y-%m")
    monthly = []
    for ym, x in h.groupby("ym"):
        monthly.append({"ym": ym, "days": len(x), "good": int(x["good"].sum()),
                        "all": _pay(x["day_diff"].sum(), x["day_inn"].sum()),
                        "t1": _pay(x["t1_d"].sum(), x["t1_i"].sum()), "t3": _pay(x["t3_d"].sum(), x["t3_i"].sum())})
    buckets = []
    allp = _pay(h["day_diff"].sum(), h["day_inn"].sum())
    for a, b in RANK_BUCKETS:
        p = _pay(h[f"b{a}_d"].sum(), h[f"b{a}_i"].sum())
        buckets.append({"from": a, "to": b, "pay": p, "lift": p - allp})
    recent = h[h["date"] >= end - pd.Timedelta(days=91)]
    r_lift = _pay(recent["t3_d"].sum(), recent["t3_i"].sum()) - _pay(recent["day_diff"].sum(), recent["day_inn"].sum())
    buckets = smooth_buckets(buckets)
    return {"strategies": strat, "monthly": monthly, "buckets": buckets, "all_pay": allp,
            "first": h["date"].min().date().isoformat(), "last": end.date().isoformat(), "days": len(h),
            "recent_t3_lift": r_lift}


def smooth_buckets(buckets: list[dict]) -> list[dict]:
    """順位ごとの出率を、上の順位ほど高い（同じ以上）形にならす（台数で重み付けした単調回帰）。"""
    if not buckets:
        return []
    vals = [b["pay"] for b in buckets]
    wts = [min(b["to"], 63) - b["from"] + 1 for b in buckets]
    blocks = [[v, w, [i]] for i, (v, w) in enumerate(zip(vals, wts))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:          # 下の順位の方が高い → まとめて平均
            v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [v, blocks[i][1] + blocks[i + 1][1], blocks[i][2] + blocks[i + 1][2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = [dict(b) for b in buckets]
    for v, _, idx in blocks:
        for j in idx:
            out[j]["pay_s"] = v
    return out


def bucket_of(rank: int) -> tuple[int, int]:
    for a, b in RANK_BUCKETS:
        if a <= rank <= b:
            return a, b
    return RANK_BUCKETS[-1]


def habits(d: pd.DataFrame) -> dict:
    """この店の癖（いまのデータから毎回計算）: 凹み台の戻り方の推移、機種ごとの出率の推移。"""
    o = d[d["obs"]].copy()
    if o.empty:
        return {}
    end = o["date"].max()
    out = {"rebound": [], "models": []}
    for name, a, b in (("1年前〜半年前", end - pd.Timedelta(days=365), end - pd.Timedelta(days=182)),
                       ("半年前〜3か月前", end - pd.Timedelta(days=182), end - pd.Timedelta(days=91)),
                       ("直近3か月", end - pd.Timedelta(days=91), end + pd.Timedelta(days=1))):
        x = o[(o["date"] >= a) & (o["date"] < b)].copy()
        if x["date"].nunique() < 20:
            continue
        x["q"] = x.groupby("date")["sd7"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
        allp = _pay(x["diff"].sum(), x["inn"].sum())
        q = [(_pay(g["diff"].sum(), g["inn"].sum()) - allp) for _, g in x.groupby("q")]
        out["rebound"].append({"period": name, "q": q, "days": int(x["date"].nunique())})
        for m, g in x.groupby("model"):
            if g["date"].nunique() >= 20:
                out["models"].append({"period": name, "model": m, "pay": _pay(g["diff"].sum(), g["inn"].sum()),
                                      "units": int(g["unit"].nunique())})
    return out
