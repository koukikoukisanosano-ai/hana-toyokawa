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
    payout: tuple[float, ...]   # 設定1..6 の機械割(%)
    verified: bool = True

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
    ),
    # スターハナハナ-30 公表値（REGが重い・BIGも軽め）
    "スターハナハナ": Spec(
        name="スターハナハナ-30",
        bb=(270, 262, 252, 240, 229, 218),
        rb=(387, 354, 322, 293, 267, 242),
        payout=(97, 99, 101, 104, 107, 110),
    ),
    # ドラゴンハナハナ～閃光～-30 / スマート沖スロ版 公表値（BIGが軽くREGが重い）
    "閃光": Spec(
        name="ドラゴンハナハナ～閃光～",
        bb=(256, 246, 235, 224, 212, 199),
        rb=(642, 585, 537, 489, 442, 399),
        payout=(97, 99, 101, 104, 107, 110),
    ),
    # キングハナハナ-30（6号機）公表値
    "キングハナハナ": Spec(
        name="キングハナハナ-30",
        bb=(292, 280, 268, 257, 244, 232),
        rb=(489, 452, 420, 390, 360, 332),
        payout=(97, 99, 101, 104, 107, 110),
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
