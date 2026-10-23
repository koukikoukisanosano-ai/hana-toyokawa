"""台の入れ替え・台番号の付け直しの見つけ方のテスト。 python tests/test_layout.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hana import layout as L  # noqa: E402

K, N = "キングハナハナ-30", "ニューキングハナハナV‐30"


def make(days, layout_of):
    rows = []
    for i, d in enumerate(pd.date_range("2026-01-01", periods=days)):
        for unit, model in layout_of(i):
            rows.append({"date": d.strftime("%Y-%m-%d"), "unit": unit, "model": model, "games": 3000, "bb": 10, "rb": 8})
    return pd.DataFrame(rows)


def base(i):
    return [(u, K) for u in range(869, 879)] + [(u, N) for u in range(927, 937)]


def test_no_change():
    u = make(30, base)
    out, ev = L.renumber(u)
    assert ev == [] and (out["unit"] == u["unit"]).all()
    assert L.model_changes(u) == []


def test_shift_renumber_with_missing_day():
    def lay(i):
        x = base(i)
        if i >= 20:
            x = [(u + 100, m) for u, m in x]
            if i == 20:
                x = x[1:]          # 付け直しの日に1台だけ取得もれ
        return x
    u = make(30, lay)
    out, ev = L.renumber(u)
    assert len(ev) == 1 and ev[0]["kind"] == "renumber" and ev[0]["linked"] == 20, ev
    early = out[pd.to_datetime(out["date"]) < "2026-01-21"]
    assert set(early["unit"]) == {u + 100 for u, _ in base(0)}


def test_order_renumber():
    def lay(i):
        x = base(i)
        if i >= 15:
            x = [(u + 1000 if u < 900 else u + 2000, m) for u, m in x]
        return x
    u = make(25, lay)
    out, ev = L.renumber(u)
    assert ev[0]["kind"] == "renumber" and ev[0]["linked"] == 20


def test_added_and_swapped():
    def lay(i):
        x = base(i)
        if i >= 10:
            x = [(u, N if 869 <= u <= 871 else m) for u, m in x] + [(1001, K), (1002, K)]
        return x
    u = make(15, lay)
    out, ev = L.renumber(u)
    assert len(ev) == 1 and ev[0]["kind"] == "added"
    ch = L.model_changes(u)
    assert len(ch) == 1 and ch[0]["items"] == [{"from": 869, "to": 871, "old": "キングハナハナ", "new": "ニューキングハナハナV"}]
    notes = L.notes(u, "ハナハナ", "2026-01-16")
    assert any("増えました" in s for s in notes) and any("入れ替え" in s for s in notes), notes


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
