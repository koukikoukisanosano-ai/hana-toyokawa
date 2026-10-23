"""ハナハナシリーズの設定別スペック。

数値はメーカー公表値（ボーナス確率の分母）。新しいハナハナ機種が店に入ったら
ここに追記すると、その機種も設定推定の対象になります。
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Spec:
    name: str
    bb: tuple[float | None, ...]   # 設定1..6 のBIG確率の分母（その設定が無い機種は None）
    rb: tuple[float | None, ...]   # 設定1..6 のREG確率の分母
    payout: tuple[float | None, ...]   # 設定1..6 の機械割(%)
    verified: bool = True
    big_coins: float | None = None     # BIG 1回の獲得枚数
    reg_coins: float | None = None     # REG 1回の獲得枚数
    coin_mochi: float | None = None    # コイン持ち（50枚で回せるG数）

    @property
    def settings(self) -> tuple[int, ...]:
        return tuple(i + 1 for i, v in enumerate(self.bb) if v is not None)


# キー: 機種名に含まれる文字列（正規化後）
SPECS: dict[str, Spec] = {
    # 並び順が大事: 名前が他の機種名に含まれるもの（キングハナハナ）は後ろに置く
    # ハナハナホウオウ～天翔～-30 公表値
    "ハナハナホウオウ~天翔~": Spec(
        name="ハナハナホウオウ～天翔～-30",
        bb=(297, 284, 273, 262, 249, 236),
        rb=(496, 458, 425, 397, 366, 337),
        payout=(97, 99, 101, 104, 107, 110),
    ),
    # ニューキングハナハナV-30 公表値。設定は 1・2・3・4・V の5段階（V を設定6の枠で扱う）
    "ニューキングハナハナV": Spec(
        name="ニューキングハナハナV-30",
        bb=(299, 291, 281, 268, None, 253),
        rb=(496, 471, 442, 409, None, 372),
        payout=(97, 99, 101, 104, None, 108),
        big_coins=312, reg_coins=130, coin_mochi=37.0,   # BIGはBT込み約312枚・REG約130枚・50枚約37G
    ),
    # スターハナハナ-30 公表値（REGが重い・BIGも軽め）
    "スターハナハナ": Spec(
        name="スターハナハナ-30",
        bb=(270, 262, 252, 240, 229, 218),
        rb=(387, 354, 322, 293, 267, 242),
        payout=(97, 99, 101, 104, 107, 110),
        big_coins=240, reg_coins=96,                     # BIG最大240枚・REG最大96枚（コイン持ちは機械割から逆算）
    ),
    # ドラゴンハナハナ～閃光～-30 / スマート沖スロ版 公表値（BIGが軽くREGが重い）
    "閃光": Spec(
        name="ドラゴンハナハナ～閃光～",
        bb=(256, 246, 235, 224, 212, 199),
        rb=(642, 585, 537, 489, 442, 399),
        payout=(97, 99, 101, 104, 107, 110),
        big_coins=252, reg_coins=96, coin_mochi=39.9,    # BIG最大252枚・REG最大96枚・50枚約39.9G
    ),
    # キングハナハナ-30（6号機）公表値
    "キングハナハナ": Spec(
        name="キングハナハナ-30",
        bb=(292, 280, 268, 257, 244, 232),
        rb=(489, 452, 420, 390, 360, 332),
        payout=(97, 99, 101, 104, 107, 110),
        big_coins=260, reg_coins=120, coin_mochi=39.9,   # BIG最大260枚・REG最大120枚・50枚約39.9G
    ),
}

# 未登録のハナハナ機種に使う仮スペック（天翔と同等と仮定）。結果ページで警告表示されます。
GENERIC = Spec(
    name="ハナハナ(スペック未登録)",
    bb=(297, 284, 273, 262, 249, 236),
    rb=(496, 458, 425, 397, 366, 337),
    payout=(97, 99, 101, 103, 106, 109),
    verified=False,
)


def normalize(name: str) -> str:
    s = unicodedata.normalize("NFKC", name or "")
    s = s.replace("〜", "~").replace("～", "~").replace(" ", "").replace("　", "")
    return s


def spec_for(model: str) -> Spec:
    n = normalize(model)
    for key, spec in SPECS.items():
        if normalize(key) in n:
            return spec
    return GENERIC


# この店の設定配分（機種ごと）。data/setting_prior.json があればそれを使う（毎週データから推定し直す）。
DEFAULT_WEIGHTS = (0.15, 0.30, 0.35, 0.15, 0.03, 0.02)    # ファイルが無いときの仮の配分（設定1〜4中心）
_PRIOR_CACHE: dict | None = None


def _prior_file() -> dict:
    global _PRIOR_CACHE
    if _PRIOR_CACHE is None:
        import json
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "data" / "setting_prior.json"
        try:
            _PRIOR_CACHE = json.loads(p.read_text(encoding="utf-8")).get("weights", {})
        except Exception:
            _PRIOR_CACHE = {}
    return _PRIOR_CACHE


def reload_prior() -> None:
    global _PRIOR_CACHE
    _PRIOR_CACHE = None


def setting_weights(sp: Spec) -> list[float]:
    """設定1〜6の配分（その機種に無い設定は0）。"""
    w = _prior_file().get(sp.name)
    if not w or len(w) != len(sp.bb):
        w = list(DEFAULT_WEIGHTS[:len(sp.bb)])
    w = [x if b else 0.0 for x, b in zip(w, sp.bb)]
    t = sum(w) or 1.0
    return [x / t for x in w]


def _coin_use_by_setting(sp: Spec) -> list[float | None]:
    """設定ごとの1Gあたりの実質消費 = BIG枚数÷BIG分母 ＋ REG枚数÷REG分母 − 3×(機械割−1)。"""
    if not (sp.big_coins and sp.reg_coins):
        return [None] * len(sp.bb)
    return [sp.big_coins / b + sp.reg_coins / r - 3 * (p / 100 - 1) if b and r and p else None
            for b, r, p in zip(sp.bb, sp.rb, sp.payout)]


def coin_use_per_game(sp: Spec) -> float | None:
    """店の設定配分で重み付けした、1Gあたりの実質消費（小役の戻りを差し引いた枚数）。
    以前は設定1〜6の単純平均を使っていたが、設定1〜4中心の店では出率を0.2〜0.5%高く見積もってしまうため改めた。"""
    cs = _coin_use_by_setting(sp)
    if all(c is None for c in cs):
        return 50.0 / sp.coin_mochi if sp.coin_mochi else None
    w = setting_weights(sp)
    pairs = [(x, c) for x, c in zip(w, cs) if c is not None]
    t = sum(x for x, _ in pairs) or 1.0
    return sum(x * c for x, c in pairs) / t


def coin_use_given(sp: Spec, bb, rb, games) -> float | None:
    """その台・その日の BIG・REG 回数から「どの設定だったか」の確率を出し、その設定に合った消費で平均する。"""
    import math
    cs = _coin_use_by_setting(sp)
    if all(c is None for c in cs) or not games or games <= 0:
        return coin_use_per_game(sp)
    w = setting_weights(sp)
    logs = []
    for x, b, r, c in zip(w, sp.bb, sp.rb, cs):
        if not b or c is None or x <= 0:
            logs.append(None)
            continue
        lb, lr = games / b, games / r
        logs.append(math.log(x) + bb * math.log(lb) - lb + rb * math.log(lr) - lr)
    m = max(v for v in logs if v is not None)
    ps = [math.exp(v - m) if v is not None else 0.0 for v in logs]
    t = sum(ps)
    return sum(p * c for p, c in zip(ps, cs) if c is not None) / t
