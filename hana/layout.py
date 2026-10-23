"""台の入れ替え（同じ台番号で機種が変わる）と、台番号の付け直しを見つける。

入れ替え
  同じ台番号で前の営業日と機種が違えば「入れ替え」。予想システムは、台番号で同じ機種が続いている期間
  （sysmodel の stint）ごとに台の記録をまとめるので、前の機種の記録が新しい台に混ざることはない。
  入れ替わった直後の台は記録が少ないぶん、機種・日の種類の平均寄りの予想になる。

台番号の付け直し
  ある日に、それまで一度も出てこなかった台番号がまとまって現れたら「付け直し」とみなす。
  消えた番号と現れた番号の数が同じで、番号順に並べた機種の並びも同じなら、番号順に対応させて
  過去の記録を新しい番号に引き継ぐ（それ以外は引き継がず、ページでお知らせする）。
"""
from __future__ import annotations

import pandas as pd


def short(m) -> str:
    return str(m).split("‐")[0].split("-")[0]


def _ranges(nums) -> list[tuple[int, int]]:
    out, s, p = [], None, None
    for x in sorted(int(v) for v in nums):
        if s is None:
            s = p = x
        elif x == p + 1:
            p = x
        else:
            out.append((s, p))
            s = p = x
    if s is not None:
        out.append((s, p))
    return out


def range_text(a: int, b: int) -> str:
    return f"{a}" if a == b else f"{a}〜{b}"


def model_changes(units: pd.DataFrame, keyword: str = "ハナハナ") -> list[dict]:
    """同じ台番号で機種が変わった日の一覧（古い順）。日付は新しい機種が最初にデータに出た日。"""
    d = units[units["model"].str.contains(keyword, na=False)][["date", "unit", "model"]].copy()
    if d.empty:
        return []
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["unit", "date"]).reset_index(drop=True)
    prev = d.groupby("unit")["model"].shift(1)
    ch = d[prev.notna() & (d["model"] != prev)].assign(old=prev)
    events = []
    for dt, g in ch.groupby("date"):
        items = []
        for (o, n), gg in g.groupby(["old", "model"]):
            for a, b in _ranges(gg["unit"]):
                items.append({"from": a, "to": b, "old": short(o), "new": short(n)})
        items.sort(key=lambda x: x["from"])
        events.append({"date": dt.date().isoformat(), "n": int(len(g)), "items": items})
    return events


def renumber(units: pd.DataFrame, keyword: str = "ハナハナ") -> tuple[pd.DataFrame, list[dict]]:
    """新しい台番号が現れた日を見つけ、付け直しと確かめられるときは過去の記録を新しい番号に書き換える。
    付け直しとみなす条件: その日に消えた番号（以後も出てこない）と現れた番号の数が同じで、番号順の機種の並びも同じ。
    返り値: (書き換えた units, 見つけた日の一覧 [{date, kind: renumber/added/changed, ...}])"""
    is_t = units["model"].str.contains(keyword, na=False)
    d = units[is_t][["date", "unit", "model"]].copy()
    if d.empty:
        return units, []
    d["date"] = pd.to_datetime(d["date"])
    d["unit"] = d["unit"].astype(int)
    d = d.sort_values("date").reset_index(drop=True)
    first = d.groupby("unit")["date"].min()
    last = d.groupby("unit")["date"].max()
    model_first = d.groupby("unit")["model"].first().map(short)    # その番号で最初に出た機種
    model_last = d.groupby("unit")["model"].last().map(short)      # その番号で最後に出た機種
    start = d["date"].min()
    win = pd.Timedelta(days=7)      # 取得もれの日があっても拾えるよう、前後1週間をまとめて見る
    done: set[int] = set()
    events, out = [], units.copy()
    out_dates = pd.to_datetime(out["date"])
    for day in sorted(first[first > start].unique()):
        new = sorted(u for u in first.index if day <= first[u] < day + win and u not in done)
        if not new:
            continue
        done |= set(new)
        gone = sorted(u for u in last.index if day - win <= last[u] < day)   # このころ消えて、その後は出てこない番号
        ev = {"date": pd.Timestamp(day).date().isoformat(), "new": len(new), "gone": len(gone),
              "new_ranges": _ranges(new), "gone_ranges": _ranges(gone)}
        mp = {}
        if gone and len(gone) == len(new) and [model_last[u] for u in gone] == [model_first[u] for u in new]:
            mp = dict(zip(gone, new))                       # 数も機種の並びも同じ → 番号順に対応
        elif gone:
            # 一律にずらした付け直し（例: 全部 +100）なら、数が少し合わなくても対応がつく
            ks = pd.Series([n - g for g in gone for n in new]).value_counts()
            if len(ks):
                k = int(ks.index[0])
                pairs = {g: g + k for g in gone if g + k in new and model_last[g] == model_first[g + k]}
                if len(pairs) >= max(3, 0.8 * min(len(gone), len(new))):
                    mp = pairs
        if mp:
            before = (out_dates < day) & is_t
            out.loc[before, "unit"] = out.loc[before, "unit"].map(lambda u: mp.get(int(u), u))
            ev["kind"] = "renumber"
            ev["linked"] = len(mp)
        elif not gone:
            ev["kind"] = "added"
        else:
            ev["kind"] = "changed"
        events.append(ev)
    return out, events


def notes(units: pd.DataFrame, keyword: str, target, recent_days: int = 45) -> list[str]:
    """ページに出すお知らせ（最近の入れ替え・台番号の付け直し）。"""
    t = pd.Timestamp(target)
    out = []
    mapped = units
    try:
        mapped, rn = renumber(units, keyword)
        for ev in rn:
            if (t - pd.Timestamp(ev["date"])).days > recent_days:
                continue
            md = ev["date"][5:].replace("-", "/")
            nr = "、".join(range_text(a, b) for a, b in ev["new_ranges"])
            gr = "、".join(range_text(a, b) for a, b in ev["gone_ranges"])
            if ev["kind"] == "renumber":
                out.append(f"{md} から台番号が付け直されたようです（{gr} → {nr}）。"
                           "機種の並びが前と同じなので、番号順に前の台の記録を引き継いで予想しています。")
            elif ev["kind"] == "added":
                out.append(f"{md} にハナハナの台が増えました（{nr}）。新しい台は記録が無いため、しばらくは機種・日の種類の平均寄りの予想です。")
            else:
                out.append(f"{md} から台番号が変わった台があります（なくなった番号 {gr}・新しい番号 {nr}）。"
                           "前の番号との対応が分からないため、新しい番号の台は記録が無い扱いです（しばらくは機種・日の平均寄りの予想）。")
    except Exception:
        pass
    for ev in model_changes(mapped, keyword):
        if (t - pd.Timestamp(ev["date"])).days > recent_days:
            continue
        parts = [f"{range_text(x['from'], x['to'])} {x['old']}→{x['new']}" for x in ev["items"]]
        out.append(f"{ev['date'][5:].replace('-', '/')} に台の入れ替えがありました（{'、'.join(parts)}）。"
                   "入れ替わった台は前の機種の記録を使わず、新しい機種になってからの記録だけで予想します。"
                   "記録が少ないうちは、機種・日の種類の平均寄りの予想になります。")
    return out
