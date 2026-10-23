"""予想結果を1枚のHTMLにする（output/latest.html と output/日付.html）。"""
from __future__ import annotations

import html
from datetime import date as Date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import model as M

E = html.escape

CSS = """
.dstr{border-radius:14px;padding:14px 16px;margin:14px 0;border:1px solid var(--line,#ddd)}
.dstr.hot{background:#fde3d8}.dstr.warm{background:#fdf0dc}.dstr.norm{background:#f2f2ef}.dstr.cold{background:#e7edf3}
.dstr b.lv{font-size:22px;margin-right:6px}.dstr .sub{font-size:13px;color:#555;margin-top:6px}
.week{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:4px;margin-top:10px}
.week div{border-radius:8px;padding:6px 2px;text-align:center;font-size:12px;color:#222}
.week .hot{background:#f3a87c}.week .warm{background:#f6dcae}.week .norm{background:#ece9e1}.week .cold{background:#dfe7ef}
table.jgrid td.jg{text-align:center;min-width:26px;font-weight:600;color:#222}
table.jgrid th{white-space:nowrap}
:root{--bg:#f6f4ef;--card:#fff;--ink:#1d1b18;--sub:#6b665d;--line:#e4dfd4;--accent:#c8324b;--accent2:#e98a2b;
--good:#1f8a5b;--bad:#9a948a;--heat0:#f1ede4;--heat1:#f9d7c4;--heat2:#f4a98a;--heat3:#e66a5a;--heat4:#c8324b;--warn:#fff4dc;--warnline:#e9c46a}
@media (prefers-color-scheme:dark){:root{--bg:#141311;--card:#1e1c19;--ink:#efebe3;--sub:#a59f94;--line:#34312b;--accent:#ff6b81;
--accent2:#ffad5c;--good:#4cc28c;--bad:#7d776d;--heat0:#26231f;--heat1:#4a2f27;--heat2:#7a3a2c;--heat3:#b04838;--heat4:#e2566a;--warn:#2b2516;--warnline:#8a6d2a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:"Hiragino Sans","Yu Gothic UI","Meiryo",system-ui,sans-serif;font-size:15px;line-height:1.55}
.wrap{max-width:1180px;margin:0 auto;padding:22px 18px 60px}
header{display:flex;flex-wrap:wrap;align-items:flex-end;gap:10px 22px;border-bottom:3px solid var(--ink);padding-bottom:12px}
h1{font-size:26px;margin:0;letter-spacing:.02em}h1 small{font-size:14px;color:var(--sub);font-weight:500;margin-left:8px}
.date{font-size:20px;font-weight:700}.tag{display:inline-block;background:var(--accent);color:#fff;border-radius:3px;padding:1px 8px;font-size:13px;margin-left:6px}
.status{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:16px 0}
.box{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 14px}
.box b{display:block;font-size:22px}.box span{color:var(--sub);font-size:13px}
.warn{background:var(--warn);border:1px solid var(--warnline);border-radius:6px;padding:10px 14px;margin:10px 0;font-size:14px}
h2{font-size:18px;margin:30px 0 10px;padding-left:10px;border-left:5px solid var(--accent)}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line)}
th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}
th{font-size:12px;color:var(--sub);font-weight:600;background:var(--bg);position:sticky;top:0}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr.top td{background:color-mix(in srgb,var(--accent) 7%,var(--card))}
.rank{font-weight:800;font-size:18px;width:42px;text-align:center}.unit{font-weight:800;font-size:20px;letter-spacing:.02em}
.bar{position:relative;height:10px;background:var(--heat0);border-radius:5px;min-width:110px;overflow:hidden}
.bar i{position:absolute;left:0;top:0;bottom:0;background:linear-gradient(90deg,var(--accent2),var(--accent));border-radius:5px}
.calc{border:1px solid var(--line,#ddd);border-radius:12px;padding:12px;margin:8px 0;background:var(--card,#fff)}
.qgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px;margin:6px 0}
.qgrid label{display:flex;flex-direction:column;font-size:12px;color:var(--sub)}
.qgrid input,.qgrid select{font-size:16px;padding:6px 8px;border:1px solid #ccc;border-radius:8px;margin-top:2px;background:#fff;color:#222;width:100%;box-sizing:border-box}
.qout{margin-top:10px}.qverdict{display:inline-block;font-size:22px;font-weight:800;padding:4px 14px;border-radius:10px;color:#fff}
.q_go{background:#1f8a5b}.q_move{background:#d98a1c}.q_stop{background:#c8324b}
.qwhy{margin:6px 0;font-weight:600}.qline{margin:4px 0}.qsub{font-size:12px;color:var(--sub)}
details summary{cursor:pointer;margin:6px 0;color:var(--sub)}
.pct{font-weight:700;font-size:16px}.pct small{font-size:11px;font-weight:400;margin-left:1px}.mark{font-size:18px}.ratio{color:var(--sub);font-size:12px}
.chip{display:inline-block;border-radius:3px;padding:0 6px;margin:2px 4px 2px 0;font-size:12px;border:1px solid var(--line)}
.chip.p{border-color:var(--good);color:var(--good)}.chip.m{color:var(--bad)}
.prev{font-size:12px;color:var(--sub);white-space:normal}.prev b{color:var(--ink);font-weight:600}
.map{display:flex;flex-direction:column;gap:12px}.island{display:flex;flex-wrap:wrap;gap:4px}
.cell{width:56px;height:56px;border-radius:5px;display:flex;flex-direction:column;align-items:center;justify-content:center;
font-weight:700;border:1px solid var(--line);font-size:15px}.cell small{font-size:11px;font-weight:500}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}
.mini td,.mini th{padding:5px 8px;font-size:13px}.hi{color:var(--accent);font-weight:700}
.note{color:var(--sub);font-size:13px}footer{margin-top:36px;color:var(--sub);font-size:12px;border-top:1px solid var(--line);padding-top:12px}
svg.spark{display:block}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
@media (max-width:700px){
.wrap{padding:14px 12px 40px}
header{gap:6px 12px}h1{font-size:19px}h1 small{display:block;margin:2px 0 0}.date{font-size:17px}
.status{grid-template-columns:repeat(2,1fr);gap:8px}.box{padding:8px 10px}.box b{font-size:19px}.box span{font-size:12px}
h2{font-size:17px;margin:24px 0 8px}
table.rk,table.rk tbody{display:block;border:none;background:none}
table.rk tr.hd{display:none}
table.rk tr{display:grid;grid-template-columns:30px 62px 1fr;gap:4px 8px;align-items:center;background:var(--card);
border:1px solid var(--line);border-radius:8px;padding:10px 10px;margin-bottom:8px}
table.rk tr.top{border-color:color-mix(in srgb,var(--accent) 45%,var(--line))}
table.rk td{display:block;border:none;padding:0;background:none!important}
table.rk td.rank{font-size:16px;width:auto}table.rk td.unit{font-size:22px}
table.rk td.reasons,table.rk td.prevc{grid-column:1/-1}
table.rk td.sparkc{display:none}
.bar{min-width:0}
.cell{width:46px;height:46px;font-size:13px}
.grid2{grid-template-columns:1fr}
.mini td,.mini th{padding:4px 6px;font-size:12px}
}
"""


def pred_value_text(target, mth, v) -> str:
    """選ばれた中身の予想値を、読める形にする。"""
    if v is None or pd.isna(v):
        return ""
    if mth == "ensemble":
        return f"予想の順位 上位{(1 - float(v)) * 100:.0f}%"
    v = float(v)
    return {"t_rank": f"予想の差枚順位 上位{(1 - v) * 100:.0f}%", "t_diff": f"予想差枚 {v:+,.0f}枚",
            "t_plus": f"プラス終了の見込み {v * 100:.0f}%", "t_payout": f"予想出率 {100 + v * 100:.1f}%",
            "t_eset": f"予想の推定設定 {v:.2f}"}.get(target, "")


# ---------------------------------------------------------------- 予想システム（sysmodel）用の表示
def _pc(x) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 + x * 100:.2f}%"


def _coins(x) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 24000:+,.0f}枚"


def _se_txt(se) -> str:
    if se is None or (isinstance(se, float) and np.isnan(se)):
        return ""
    return f"<small> ±{se * 100:.2f}</small>"


def sys_bucket_text(info: dict, rank: int) -> str:
    """その順位の台が、過去の答え合わせで実際どれだけ出たか。"""
    sm = info.get("summary") or {}
    b = next((x for x in sm.get("buckets", []) if x["from"] <= rank <= x["to"]), None)
    if not b:
        return ""
    pay = b.get("pay_s", b["pay"])
    lift = pay - sm.get("all_pay", pay - b["lift"])
    rng = f"{b['from']}位" if b["from"] == b["to"] else f"{b['from']}〜{min(b['to'], 99)}位"
    return f"過去の{rng}の台：出率 {_pc(pay)}（全台平均より{lift * 100:+.2f}%・8000Gで{pay * 24000:+,.0f}枚）"


def sys_note(info: dict, n_units: int) -> str:
    sm = info.get("summary") or {}
    st = next((x for x in sm.get("strategies", []) if x["period"] == "全期間"), None)
    head = (f"全{n_units}台を、今日の出率（払い出し÷投入）が高そうな順に並べています。予想は機械学習（勾配ブースティング木）で、"
            "直近の差枚（凹み・出すぎ）、前日の結果、機種、日の種類などから学習し、2週間ごとに学び直します。")
    if st:
        r = {x["label"]: x for x in st["rows"]}
        a, t3 = r.get("毎日行く・適当な台（全台平均）"), r.get("毎日行く・予想上位3台のどれか")
        if a and t3:
            head += (f"過去{st['days']}日の答え合わせでは、予想上位3台の出率は {_pc(t3['pay'])}"
                     f"（全台平均 {_pc(a['pay'])}・8000G打った場合 {_coins(t3['pay'])} 対 {_coins(a['pay'])}）。"
                     "1台だけ選ぶと、どの台になるかの小さな違いで結果が大きくぶれるので、上位3台の中から空いている台を選ぶのがおすすめです。")
    return head + "点数は今日の中での位置（1位＝100点）。詳しくは下の「2年間の答え合わせ」。"


def sys_answer_section(info: dict) -> str:
    sm = info.get("summary") or {}
    if not sm.get("strategies"):
        return ""
    blocks = []
    for st in sm["strategies"]:
        rows = "".join(
            f"<tr{' style=\'font-weight:700\'' if '強い日だけ行く・予想1位' in x['label'] else ''}><td>{E(x['label'])}</td>"
            f"<td class='num'>{_pc(x['pay'])}{_se_txt(x.get('se'))}</td>"
            f"<td class='num'>{_coins(x['pay'])}</td><td class='num'>{x['days']}日</td></tr>" for x in st["rows"])
        blocks.append(f"<div><h3 style='font-size:15px;margin:6px 0'>{E(st['period'])}（{st['since']}〜・{st['days']}日）</h3>"
                      f"<table class='mini'><tr><th>やり方</th><th>出率</th><th>1日8000Gの平均</th><th>日数</th></tr>{rows}</table></div>")
    mrows = "".join(
        f"<tr><td>{m['ym']}</td><td class='num'>{_pc(m['all'])}</td><td class='num'>{_pc(m['t1'])}</td>"
        f"<td class='num'>{_pc(m['t3'])}</td><td class='num'>{m['good']}/{m['days']}日</td></tr>"
        for m in reversed(sm.get("monthly", [])))
    brows = "".join(
        f"<tr><td>{b['from']}位" + ("" if b['from'] == b['to'] else f"〜{min(b['to'], 99)}位") + "</td>"
        f"<td class='num'>{_pc(b['pay'])}</td><td class='num'>{_pc(b.get('pay_s', b['pay']))}</td>"
        f"<td class='num'>{b.get('pay_s', b['pay']) * 24000:+,.0f}枚</td></tr>"
        for b in sm.get("buckets", []))
    warn = ""
    if sm.get("recent_t3_lift") is not None and sm["recent_t3_lift"] < 0:
        warn = ("<div class='warn'>直近3か月は、予想上位3台の出率が全台平均を下回っています。店のやり方が変わった可能性があります"
                "（予想は2週間ごとに学び直すので、変化に合わせて変わっていきます）。</div>")
    return (f"<h2>2年間の答え合わせ</h2>{warn}<p class='note'>{sm.get('first')}〜{sm.get('last')}の各日を、"
            "<b>その日より前のデータだけ</b>で予想し直し、実際の結果と比べました（学習は2週間ごとにやり直し）。"
            "出率はBIG・REG回数とG数から計算しているので、他の人の打ち方（やめ方）の影響を受けにくい数字です。"
            "「強い日」は、その日の時点で予想した店全体の出率が、それまでの予想の真ん中より上の日。"
            f"±は偶然の幅のめやす（{sm.get('when', '')}時点・週1回作り直し）。</p>"
            f"<div class='grid2'>{''.join(blocks)}</div>"
            f"<div class='grid2'><div><h3 style='font-size:15px;margin:6px 0'>予想の順位ごとの実際の出率</h3>"
            f"<table class='mini'><tr><th>予想の順位</th><th>出率</th><th>ならした出率</th><th>8000G換算</th></tr>{brows}</table>"
            "<p class='note'>「ならした出率」は、1位と2〜3位の逆転のような偶然の凸凹をならし、上の順位ほど高くなるようにした値。</p></div>"
            f"<div><h3 style='font-size:15px;margin:6px 0'>月ごとの成績</h3><div class='scroll'><table class='mini'>"
            f"<tr><th>月</th><th>全台平均</th><th>予想1位</th><th>予想上位3台</th><th>強い日</th></tr>{mrows}</table></div></div></div>")


def sys_research_section(info: dict) -> str:
    rs = info.get("research") or {}
    if not rs.get("units"):
        return ""
    rows = "".join(
        f"<tr{' style=\'font-weight:700\'' if u['method'] == rs.get('chosen') else ''}><td>{E(u['label'])}"
        f"{' ★採用' if u['method'] == rs.get('chosen') else ''}</td>"
        f"<td class='num'>{u['dev'] * 100:+.2f}%</td><td class='num'>{u['hold'] * 100:+.2f}%</td>"
        f"<td class='num'>{u['all'] * 100:+.2f}%</td><td class='num'>{u['q_pos']}/{u['q_n']}</td></tr>"
        for u in sorted(rs["units"], key=lambda u: -u["hold"]))
    drows = "".join(
        f"<tr{' style=\'font-weight:700\'' if x['method'] == rs.get('day_chosen') else ''}><td>{E(x['label'])}"
        f"{' ★採用' if x['method'] == rs.get('day_chosen') else ''}</td><td class='num'>{x['gap'] * 100:+.2f}%</td></tr>"
        for x in rs.get("days", []) if x["period"] == "all")
    return (f"<h2>予想方法の比較（{rs.get('when')}に実施）</h2><p class='note'>データ {E(rs.get('data', ''))}。{E(rs.get('rule', ''))}"
            f"開発期間 {E(rs.get('dev', ''))}／本番テスト期間 {E(rs.get('hold', ''))}。"
            "数字は予想上位1・3・5台の出率が全台平均より何%高かったかの平均（1%＝8000Gで240枚）。"
            "凹み台を狙う方法は2025年にはよく効いていましたが、2026年3月以降に店のやり方が変わって効かなくなりました。"
            "機械学習は傾向を学び直すため、変化のあとも成績を保っていました。</p>"
            f"<div class='scroll'><table class='mini'><tr><th>台の選び方</th><th>開発期間</th><th>本番テスト</th><th>全期間</th>"
            f"<th>四半期ごとにプラス</th></tr>{rows}</table></div>"
            f"<h3 style='font-size:15px;margin:10px 0 6px'>行く日の選び方（強いと予想した日と弱いと予想した日の出率の差・全期間）</h3>"
            f"<table class='mini'><tr><th>日の選び方</th><th>差</th></tr>{drows}</table>")


def sys_habits_section(info: dict) -> str:
    hb = info.get("habits") or {}
    if not hb.get("rebound"):
        return ""
    rr = "".join(f"<tr><td>{E(x['period'])}</td>" + "".join(f"<td class='num'>{v * 100:+.2f}%</td>" for v in x["q"])
                 + f"<td class='num'>{x['days']}日</td></tr>" for x in hb["rebound"])
    periods = [x["period"] for x in hb["rebound"]]
    models = sorted({x["model"] for x in hb["models"]})
    mr = "".join(
        f"<tr><td>{E(m.split('‐')[0].split('-')[0])}</td>"
        + "".join(f"<td class='num'>{_pc(next((x['pay'] for x in hb['models'] if x['model'] == m and x['period'] == pp), np.nan))}</td>"
                  for pp in periods) + "</tr>" for m in models)
    return ("<h2>この店の癖（いまのデータから毎回計算）</h2><div class='grid2'>"
            "<div><h3 style='font-size:15px;margin:6px 0'>直近7日の差枚で5つに分けた台の、次の日の出率（全台平均との差）</h3>"
            f"<div class='scroll'><table class='mini'><tr><th>期間</th><th>一番凹み</th><th>凹み</th><th>中</th><th>出</th><th>一番出</th><th>日数</th></tr>{rr}</table></div>"
            "<p class='note'>左ほど上がる形なら「凹み台を上げる」癖。2025年ははっきり出ていましたが、最近は弱まっています。</p></div>"
            "<div><h3 style='font-size:15px;margin:6px 0'>機種ごとの出率</h3>"
            f"<table class='mini'><tr><th>機種</th>{''.join(f'<th>{E(pp)}</th>' for pp in periods)}</tr>{mr}</table></div></div>")


# ---------------------------------------------------------------- 続けるか・移るか・やめるか（撤退の判断）
QUIT_JS = """
<script>
(function(){
const Q = __Q__;
const MARGIN = __M__;
const $ = id => document.getElementById(id);
const num = id => { const v = parseFloat($(id).value); return isNaN(v) ? 0 : v; };
function post(m, rank, G, B, R){
  const P = Q.models[m]; if(!P) return null;
  const bk = P.buckets.find(b => rank >= b.from && rank <= b.to) || P.buckets[P.buckets.length-1];
  let best = -1e300; const logs = [];
  for (let i = 0; i < P.pay.length; i++){
    const w = bk.w[i];
    if (!P.bb[i] || w <= 0){ logs.push(null); continue; }
    let v = Math.log(w);
    if (G > 0){ const lb = G / P.bb[i], lr = G / P.rb[i]; v += B*Math.log(lb) - lb + R*Math.log(lr) - lr; }
    logs.push(v); if (v > best) best = v;
  }
  let n = 0, d = 0;
  logs.forEach((v, i) => { if (v === null) return; const e = Math.exp(v - best); n += e * P.pay[i]; d += e; });
  let eb = 0, er = 0; bk.w.forEach((w, i) => { if (P.bb[i]) { eb += w / P.bb[i]; er += w / P.rb[i]; } });
  return {pay: n / d, prior: bk.mean, eb: eb * G, er: er * G};
}
function coins(pay, g){ return (pay - 100) / 100 * 3 * g; }
function fmt(x){ return (x >= 0 ? "+" : "") + Math.round(x).toLocaleString() + "枚"; }
function run(){
  const m = $("q_m").value, rank = Math.max(1, num("q_r")), G = num("q_g"), B = num("q_b"), R = num("q_reg");
  const rest = Math.max(0, num("q_rest"));
  const c = post(m, rank, G, B, R); if (!c) return;
  let html = "<div class='qline'>この台のこの先の見込み出率 <b>" + c.pay.toFixed(2) + "%</b>"
    + "（打つ前の見込み " + c.prior.toFixed(2) + "%）<br>あと" + rest.toLocaleString() + "G打つと平均 <b>" + fmt(coins(c.pay, rest)) + "</b></div>";
  if (G > 0) html += "<div class='qsub'>ここまで BIG " + B + "回（平均なら" + c.eb.toFixed(1) + "回）・REG " + R + "回（平均なら" + c.er.toFixed(1) + "回）</div>";
  let alt = null;
  if ($("a_r").value !== ""){
    alt = post($("a_m").value, Math.max(1, num("a_r")), num("a_g"), num("a_b"), num("a_reg"));
    if (alt) html += "<div class='qline'>移る候補の台の見込み出率 <b>" + alt.pay.toFixed(2) + "%</b>（あと" + rest.toLocaleString() + "Gで平均 " + fmt(coins(alt.pay, rest)) + "）</div>";
  }
  let verdict, cls, why;
  if (c.pay < 100 && (!alt || alt.pay < 100)){
    verdict = "やめる"; cls = "q_stop";
    why = "この先は打つほど負けの見込みです（見込み出率が100%を下回っています）。";
  } else if (alt && alt.pay - c.pay >= MARGIN){
    verdict = "移る"; cls = "q_move";
    why = "候補の台のほうが見込みが高く、あと" + rest.toLocaleString() + "Gで平均 " + fmt(coins(alt.pay, rest) - coins(c.pay, rest)) + " の差です。";
  } else {
    verdict = "続ける"; cls = "q_go";
    why = c.pay < 100 ? "この台は100%を下回りますが、候補の台がそれより良いので移りましょう。"
        : (alt && alt.pay > c.pay ? "候補の台のほうがわずかに上ですが、差が小さい（あと" + rest.toLocaleString() + "Gで " + fmt(coins(alt.pay, rest) - coins(c.pay, rest)) + "）ので、続けて問題ありません。"
        : "この先も打つほどプラスの見込みです。出だしが悪くても、この店の設定の散らばり方では、まだ設定が低いとは言えません。");
    if (c.pay < 100 && alt && alt.pay >= 100){ verdict = "移る"; cls = "q_move"; }
  }
  html = "<div class='qverdict " + cls + "'>" + verdict + "</div><div class='qwhy'>" + why + "</div>" + html;
  $("q_out").innerHTML = html;
}
document.querySelectorAll("#qcalc input, #qcalc select").forEach(e => e.addEventListener("input", run));
$("q_g").addEventListener("input", () => { if (!$("q_rest").dataset.touched) $("q_rest").value = Math.max(0, 8000 - num("q_g")); run(); });
$("q_rest").addEventListener("input", () => { $("q_rest").dataset.touched = "1"; });
run();
})();
</script>
"""


def quit_section(q: dict) -> str:
    """続けるか・移るか・やめるかの計算機と早見表。"""
    if not q or not q.get("params", {}).get("models"):
        return ""
    import json as _json
    P = q["params"]
    opts = "".join(f"<option value='{E(m)}'>{E(v['label'])}</option>" for m, v in P["models"].items())
    rows = ""
    for r in q.get("table", []):
        def cell(x):
            return "（まず起きない）" if not x else f"BIG{x[0]}・REG{x[1]}以下<small>（平均の{x[2] * 100:.0f}%）</small>"
        rows += (f"<tr><td>{E(r['model'])}</td><td class='num'>{r['G']:,}G</td><td class='num'>BIG{r['avg'][0]}・REG{r['avg'][1]}</td>"
                 f"<td>{cell(r['move'])}</td><td>{cell(r['stop'])}</td></tr>")
    rs = q.get("research") or {}
    sim = ""
    if rs.get("rows"):
        def pick(model, rank, pol, widen=1.0):
            for x in rs["rows"]:
                if x["model"] == model and x["rank"] == rank and abs(x["widen"] - widen) < 1e-9 and pol in x:
                    return x[pol]
            return None
        lines = []
        for lab in ("予想上位3台", "予想6〜10位", "予想21〜40位", "予想41位以下"):
            k_go, k_st = pick("キングハナハナ", lab, "続行"), pick("キングハナハナ", lab, "撤退")
            k_mv = pick("キングハナハナ", lab, "撤退＋移動")
            if not (k_go and k_st):
                continue
            lines.append(f"<tr><td>{E(lab)}</td><td class='num'>{k_go['coins']:+,}枚</td>"
                         f"<td class='num'>{k_st['coins']:+,}枚<small>（{k_st['diff']:+}・平均{k_st['games']:,}G・{k_st['stop'] * 100:.0f}%の日でやめる）</small></td>"
                         f"<td class='num'>{(str(format(k_mv['coins'], '+,')) + '枚<small>（' + format(k_mv['diff'], '+') + '）</small>') if k_mv else '-'}</td></tr>")
        sim = ("<h3 style='font-size:15px;margin:12px 0 6px'>基準に従うと得か（キングハナハナ・1日8000Gの計算）</h3>"
               "<div class='scroll'><table class='mini'><tr><th>座った台</th><th>最後まで打つ</th><th>見込み100%未満でやめる</th>"
               f"<th>やめる＋空き台へ移る</th></tr>{''.join(lines)}</table></div>"
               f"<p class='note'>{E(rs.get('note', ''))}（{E(rs.get('when', ''))}計算）</p>")
    js = QUIT_JS.replace("__Q__", _json.dumps(P, ensure_ascii=False)).replace("__M__", str(q.get("margin", 0.4)))
    return ("<h2>続けるか・移るか・やめるか</h2>"
            "<p class='note'>打っている台の、今日の<b>G数・BIG回数・REG回数</b>（データ機の数字）を入れると、"
            "この先の見込み出率と、続ける／移る／やめるの判断を出します。ハナハナは毎ゲームが独立した抽選で天井もないので、"
            "判断は「この台の設定がどれくらいか」の見込みだけで決まります。この店は設定1〜4の狭い範囲で動かしているため、"
            "途中の出方では設定がなかなか分からず、<b>予想上位の台なら、出だしが悪くても続けるのが基本</b>です。"
            "予想が下位の台は、少し悪いだけでやめる判断になります（そもそも座らないのが一番です）。</p>"
            "<div id='qcalc' class='calc'>"
            f"<div class='qgrid'><label>機種<select id='q_m'>{opts}</select></label>"
            "<label>この台の予想順位<input id='q_r' type='number' inputmode='numeric' value='1' min='1' max='63'></label>"
            "<label>今のG数<input id='q_g' type='number' inputmode='numeric' value='3000' min='0'></label>"
            "<label>BIG回数<input id='q_b' type='number' inputmode='numeric' value='9' min='0'></label>"
            "<label>REG回数<input id='q_reg' type='number' inputmode='numeric' value='5' min='0'></label>"
            "<label>閉店まであと何G打てるか<input id='q_rest' type='number' inputmode='numeric' value='5000' min='0'></label></div>"
            "<details><summary>空いている別の台に移るか迷うとき</summary><div class='qgrid'>"
            f"<label>候補の機種<select id='a_m'>{opts}</select></label>"
            "<label>候補の予想順位<input id='a_r' type='number' inputmode='numeric' placeholder='例: 4' min='1' max='63'></label>"
            "<label>候補の今日のG数<input id='a_g' type='number' inputmode='numeric' value='0' min='0'></label>"
            "<label>候補のBIG<input id='a_b' type='number' inputmode='numeric' value='0' min='0'></label>"
            "<label>候補のREG<input id='a_reg' type='number' inputmode='numeric' value='0' min='0'></label></div></details>"
            "<div id='q_out' class='qout'></div></div>"
            "<h3 style='font-size:15px;margin:12px 0 6px'>早見表（予想上位3台に座っているとき・BIGとREGが同じ割合で少ない場合）</h3>"
            "<div class='scroll'><table class='mini'><tr><th>機種</th><th>G数</th><th>平均なら</th>"
            f"<th>空いている4〜5位の台に移る線</th><th>やめる線</th></tr>{rows}</table></div>"
            "<p class='note'>「移る」は、移った先の見込みが0.4%以上高いとき（あと4000Gで約50枚の差）だけにしています。"
            "移れる台が無いときは、やめる線までは続けてください。数字は毎週、この店のデータから計算し直します。</p>"
            + sim + js)


GOAL_WORD = {"plus": "プラス終了", "rank": "プラス終了", "high": "高設定っぽい"}


def goal_title(cfg) -> str:
    if cfg.goal == "rank":
        return "差枚順位の予想"
    return "プラス終了の見込み" if cfg.goal == "plus" else f"高設定(設定{cfg.high_min}以上)期待度"


def factor_note(model) -> str:
    """どの要因で順位を決めているか（この店のデータで選んだものか、初期値か）。"""
    used = [M.factor_label(c) for c in model.cols if c not in M.DAY_FACTORS]
    day = "日付末尾・曜日・特定日は、その日の全台に同じだけ効く（店全体の期待台数を動かす）要因として常に使っています。"
    info = M.ACTIVE_INFO
    if info.get("source") == "selected":
        off = [M.factor_label(c) for c in M.ALL_FACTORS if c not in model.cols and c not in M.DAY_FACTORS]
        head = (f"台の順位は、この店の{info.get('n_days')}日分のデータで検証して効いた要因（{'・'.join(used)}）で決めています"
                if used else f"この店の{info.get('n_days')}日分のデータで検証したところ、台の順位に効く要因はまだ見つかっていません")
        return (head + f"（{info.get('when')} 選択・週1回とデータが2割増えるたびに自動で選び直し）。"
                + (f"効かなかったので使っていない要因: {'・'.join(off)}。" if off else "") + day)
    return (f"台の順位は、初期設定の要因（{'・'.join(used)}）で決めています。"
            "この店のデータが60日分たまると、台番・末尾・角台・前日の凹みなども含めて検証し、この店で効く要因を自動で選び直します。" + day)


def heat_class(p: float, base: float) -> str:
    r = p / base if base > 0 else 1
    return "var(--heat4)" if r >= 1.8 else "var(--heat3)" if r >= 1.4 else "var(--heat2)" if r >= 1.15 \
        else "var(--heat1)" if r >= 0.95 else "var(--heat0)"


GRADES = [(0.10, "◎", "var(--heat4)"), (0.30, "○", "var(--heat3)"), (0.70, "△", "var(--heat1)"), (1.01, "×", "var(--heat0)")]


def grade_of(rank: int, n: int) -> tuple[str, str]:
    """今日の台の中での順位 → 印と色（上位10%=◎・30%=○・70%=△・それ以下=×）。"""
    q = (rank - 0.5) / max(n, 1)
    for lim, mark, color in GRADES:
        if q < lim:
            return mark, color
    return GRADES[-1][1], GRADES[-1][2]


def spark(vals: list[float]) -> str:
    if not vals:
        return ""
    w, h = 84, 22
    xs = np.linspace(2, w - 2, len(vals)) if len(vals) > 1 else np.array([w / 2])
    ys = [h - 2 - (min(max(v, 1), 6) - 1) / 5 * (h - 4) for v in vals]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    mid = h - 2 - (3.5 - 1) / 5 * (h - 4)
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.8" fill="{"var(--accent)" if v >= 4.2 else "var(--sub)"}"/>'
                   for x, y, v in zip(xs, ys, vals))
    return (f'<svg class="spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="直近の推定設定">'
            f'<line x1="0" x2="{w}" y1="{mid:.1f}" y2="{mid:.1f}" stroke="var(--line)" stroke-dasharray="2 2"/>'
            f'<polyline points="{pts}" fill="none" stroke="var(--sub)" stroke-width="1"/>{dots}</svg>')


def fmt_int(v, sign=False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{int(v):+,}" if sign else f"{int(v):,}"


def habit_table(t: pd.DataFrame, base: float, title: str, extra: pd.DataFrame | None = None, key=None) -> str:
    if t is None or len(t) == 0:
        return ""
    rows = []
    for _, r in t.iterrows():
        ratio = r["rate"] / base if base > 0 else 1
        cls = ' class="hi"' if ratio >= 1.3 and r["days"] >= 5 else ""
        rows.append(f"<tr><td>{E(str(r['label']))}</td><td class='num'{cls}>{r['rate']*100:.1f}%</td>"
                    f"<td class='num'>×{ratio:.2f}</td><td class='num'>{fmt_int(r['avg_diff'], True)}</td>"
                    f"<td class='num'>{int(r['days'])}日</td></tr>")
    return (f"<div><h3 style='font-size:15px;margin:6px 0'>{E(title)}</h3><table class='mini'><tr><th>条件</th><th>率</th>"
            f"<th>平常比</th><th>平均差枚</th><th>日数</th></tr>{''.join(rows)}</table></div>")


def check_section(hist: pd.DataFrame, pred_dir: Path | None, target: Date, top_k: int = 3, show: int = 7) -> str:
    """過去の予想（data/predictions/*.csv）と、その後に公開された実際のデータの答え合わせ。"""
    if pred_dir is None or not Path(pred_dir).exists() or hist.empty:
        return ""
    rows = []
    for f in sorted(Path(pred_dir).glob("*.csv"), reverse=True):
        try:
            d = pd.Timestamp(f.stem)
        except ValueError:
            continue
        if d.date() >= target:
            continue
        day = hist[hist["date"] == d]
        if day.empty:
            continue
        pr = pd.read_csv(f)
        picks = pr.sort_values("rank").head(top_k)["unit"].astype(int).tolist()
        act = day.set_index("unit")
        got = [u for u in picks if u in act.index]
        if not got:
            continue
        sub = act.loc[got]
        best = set(day.sort_values("e_set", ascending=False).head(top_k)["unit"].astype(int))
        rows.append({
            "date": d.date(), "picks": got, "act": sub,
            "top_diff": float(sub["diff"].mean()), "all_diff": float(day["diff"].mean()),
            "top_e": float(sub["e_set"].mean()), "all_e": float(day["e_set"].mean()),
            "hit": len(best & set(got)),
        })
    if not rows:
        return ("<h2>予想の答え合わせ</h2><p class='note'>予想した日のデータが公開されると、ここに予想上位"
                f"{top_k}台の実際の結果が表示されます。</p>")
    n = len(rows)
    win = sum(r["top_diff"] > r["all_diff"] for r in rows)
    avg_top = np.mean([r["top_diff"] for r in rows])
    avg_all = np.mean([r["all_diff"] for r in rows])
    plus = sum(int((r["act"]["diff"] > 0).sum()) for r in rows)
    npick = sum(len(r["act"]) for r in rows)
    summary = (f"<div class='status'><div class='box'><b>{win}/{n}日</b><span>予想上位{top_k}台が全台平均を上回った日</span></div>"
               f"<div class='box'><b>{avg_top:+,.0f}枚</b><span>予想上位{top_k}台の平均差枚（全台平均 {avg_all:+,.0f}枚）</span></div>"
               f"<div class='box'><b>{plus}/{npick}台</b><span>予想上位{top_k}台のうちプラスで終わった台</span></div></div>")
    trs = []
    for r in rows[:show]:
        cells = " ".join(
            f"<span class='chip {'p' if a['diff'] > 0 else 'm'}'>{u}: {fmt_int(a['diff'], True)}枚 推定{a['e_set']:.1f}</span>"
            for u, a in r["act"].iterrows())
        trs.append(f"<tr><td class='num'>{r['date']:%m/%d}</td><td>{cells}</td>"
                   f"<td class='num'>{r['top_diff']:+,.0f}</td><td class='num'>{r['all_diff']:+,.0f}</td></tr>")
    return (f"<h2>予想の答え合わせ</h2>{summary}<div class='scroll'><table class='mini'><tr><th>日付</th><th>予想上位{top_k}台の実際</th>"
            f"<th>上位平均</th><th>全台平均</th></tr>{''.join(trs)}</table></div>"
            "<p class='note'>この仕組みが毎朝出した予想を、後から公開された実データで採点したものです（推定＝BB・REGから推定した設定）。</p>")


def day_section(hist, target, cfg) -> str:
    from . import daystrength as DS
    try:
        r = DS.estimate(hist, target, cfg.event_digits)
    except Exception:
        return ""
    if not r:
        return ""
    t = r["today"]
    lv, cls = DS.level(t["delta"])
    week = "".join(
        f"<div class='{DS.level(a['delta'])[1]}'>{a['date'].month}/{a['date'].day}<br>{DS.WEEKDAY[a['date'].weekday()]}<br>"
        f"<b>{DS.level(a['delta'])[0]}</b><br><small>{a['pay']:.1f}%</small></div>" for a in r["ahead"])
    rows = "".join(f"<tr><td>{E(x['cat'])}</td><td class='num'>{x['pay']:.2f}%</td><td class='num'>{x['exp8000']:+,.0f}枚</td>"
                   f"<td class='num'>{x['n']}日</td></tr>" for x in r["table"])
    return (f"<h2>今日の店の強さ</h2><div class='dstr {cls}'><b class='lv'>{lv}</b>"
            f"今日（{E(t['why'])}・{E(t['cat'])}）のハナハナ全体の見込み出率 {t['pay']:.2f}%"
            f"（全日平均 {r['base']:.2f}%・1台8000G打った場合の期待差枚 {t['exp8000']:+,.0f}枚）"
            f"<div class='week'>{week}</div>"
            f"<div class='sub'>混む日（土日・祝日・GW・お盆・年末年始）は出率が下がり、ただし混む日でも0・5のつく日は下がりません。"
            f"平日では火曜がやや高め。この日の選び方は、過去の各日をその日より前のデータだけで予想し直しても、"
            f"強いと予想した日が弱いと予想した日より出率が約0.4%高いことを確かめてあります（数字は過去{r['n_days']}日から毎回計算し直し）。"
            f"座る台を選ぶ効果（下のランキング）と合わせて使うと一番成績がよくなります。</div>"
            f"<table class='mini'><tr><th>日の種類</th><th>出率</th><th>8000Gの期待差枚</th><th>日数</th></tr>{rows}</table></div>")


def judge_txt(r) -> str:
    j = r.get("judge")
    if j is None or pd.isna(j):
        return "<span class='note'>G数少なく判定なし</span>"
    ga = f"合算1/{r['gassan']:.0f}" if pd.notna(r.get("gassan")) else "合算-"
    rg = f"REG1/{r['reg_den']:.0f}" if pd.notna(r.get("reg_den")) else "REG0回"
    return f"判定<b>設定{int(j)}</b>（{ga}・{rg}）"


JUDGE_BG = {1: "#e8edf3", 2: "#dfe7ef", 3: "#f1ecd9", 4: "#f6dcae", 5: "#f3a87c", 6: "#e5684f"}


def judge_grid(hist: pd.DataFrame, units: list[int], cfg, days: int = 14) -> str:
    """台×日付の判定設定の表（直近days日）。"""
    if hist is None or hist.empty or "judge" not in hist:
        return ""
    dates = sorted(hist["date"].unique())[-days:]
    h = hist[hist["date"].isin(dates)]
    look = {(int(r.unit), r.date): r for r in h.itertuples()}
    head = "".join(f"<th>{pd.Timestamp(d):%m/%d}</th>" for d in dates)
    rows = []
    for u in sorted(units):
        cells = []
        js = []
        for d in dates:
            r = look.get((u, d))
            if r is None:
                cells.append("<td class='jg'>-</td>")
                continue
            j = r.judge
            if pd.isna(j):
                cells.append(f"<td class='jg' title='{int(r.games):,}G'>·</td>")
                continue
            js.append(j)
            tip = f"{int(r.games):,}G BB{int(r.bb)} RB{int(r.rb)} 合算1/{r.gassan:.0f}" + (f" REG1/{r.reg_den:.0f}" if pd.notna(r.reg_den) else "")
            cells.append(f"<td class='jg' style='background:{JUDGE_BG[int(j)]}' title='{E(tip)}'>{int(j)}</td>")
        avg = f"{np.mean(js):.1f}" if js else "-"
        hi = sum(1 for j in js if j >= cfg.high_min)
        rows.append(f"<tr><th>{u}</th>{''.join(cells)}<td class='num'>{avg}</td><td class='num'>{hi}</td></tr>")
    from .estimate import gassan_to_setting
    tot_cells, avg_cells = [], []
    for d in dates:
        dd = h[h["date"] == d]
        if dd.empty or (dd["bb"] + dd["rb"]).sum() == 0:
            tot_cells.append("<td>-</td>"); avg_cells.append("<td>-</td>"); continue
        ga = dd["games"].sum() / (dd["bb"] + dd["rb"]).sum()
        tot_cells.append(f"<td class='jg' title='全台合算1/{ga:.0f}'><b>{gassan_to_setting(ga):.1f}</b></td>")
        jm = dd["judge"].mean()
        avg_cells.append(f"<td class='jg'>{jm:.1f}</td>" if pd.notna(jm) else "<td>-</td>")
    hh = h.dropna(subset=["judge"])
    ga_all = hh["games"].sum() / max((hh["bb"] + hh["rb"]).sum(), 1)
    rows.append(f"<tr><th>全台合算</th>{''.join(tot_cells)}<td class='num'>{gassan_to_setting(ga_all):.1f}</td><td></td></tr>")
    rows.append(f"<tr><th>判定平均</th>{''.join(avg_cells)}<td class='num'>{hh['judge'].mean():.1f}</td><td></td></tr>")
    return (f"<h2>台ごとの判定設定（直近{len(dates)}日）</h2>"
            f"<p class='note'>その日のBIG・REG（合算）から一番近い設定（差枚は使わない）。最下行は、全台を合わせた合算から出した設定と、各台の判定の平均。"
            f"{cfg.judge_min_games:,}G未満の日は判定なし（·）。1日だけの判定は運でぶれるので、何日分かまとめて見てください。"
            f"マスを長押しすると回転数とボーナス回数が出ます。</p>"
            f"<div class='scroll'><table class='mini jgrid'><tr><th>台</th>{head}<th>平均</th><th>{cfg.high_min}以上</th></tr>"
            + "".join(rows) + "</table></div>")


def build(model: M.Model, pred: pd.DataFrame, target: Date, hall_name: str, top_n: int,
          bt: dict | None, cov: dict, out_dir: Path, extra_warns: list[str] | None = None,
          pred_dir: Path | None = None, info_line: str = "") -> Path:
    cfg = model.cfg
    info = M.target_day_info(target, cfg)
    hist = model.hist
    last_date = hist["date"].max().date() if len(hist) else None
    lag = int(M.open_gap([last_date], [target], cfg)[0]) if last_date else None   # 休業日を除いた営業日数
    base = model.base_rate
    n_units = len(pred)
    exp_high = float(pred["score"].sum()) if n_units else 0.0
    avg_high = base * n_units

    sysinfo = pred.attrs.get("system")
    tags = "".join(f"<span class='tag'>{E(t)}</span>" for t in info["tags"])
    head = (f"<header><h1>{E(hall_name)} ハナハナ予想<small>{'行く日と座る台の予想' if sysinfo else goal_title(cfg)}</small></h1>"
            f"<div class='date'>{target.year}年{target.month}月{target.day}日({info['wday']}){tags}</div></header>")

    boxes = [
        (f"{n_units}台" if cfg.goal == "rank" else f"{exp_high:.1f}台", (f"今日のプラス終了 見込み台数（全{n_units}台・平常{avg_high:.1f}台）" if cfg.goal == "plus" else
         f"予想の対象台数（差枚順位の予想）" if cfg.goal == "rank" else f"今日の高設定 期待台数（全{n_units}台・平常{avg_high:.1f}台）")),
        (f"{last_date:%m/%d}" if last_date else "-", f"データの最新日（{lag}営業日前）" if lag is not None else "データなし"),
        (f"{model.n_days}日分", "学習に使ったデータ"),
        ("調整済" if model.tuned else "蓄積中", "要因の重み（45日分以上で自動調整）"),
    ]
    if sysinfo:
        st = next((x for x in (sysinfo.get("summary") or {}).get("strategies", []) if x["period"] == "全期間"), None)
        r_ = {x["label"]: x for x in st["rows"]} if st else {}
        t3, al = r_.get("毎日行く・予想上位3台のどれか"), r_.get("毎日行く・適当な台（全台平均）")
        boxes = [(f"{n_units}台", "予想の対象台数"), boxes[1], (f"{model.n_days}日分", "学習に使ったデータ")]
        if t3 and al:
            boxes.append((_coins(t3["pay"]), f"予想上位3台・8000G打った場合の平均（過去{st['days']}日の答え合わせ・全台平均{_coins(al['pay'])}）"))
    if info_line.startswith("iPhoneからの最終受信: "):
        rest = info_line.split(": ", 1)[1]
        when, _, msg = rest.partition("（")
        when = when.strip()
        if msg.endswith("）"):
            msg = msg[:-1]
        short = when[5:].replace("-", "/") if len(when) >= 16 else when
        boxes.append((short, "iPhoneから最終受信" + (f"・{msg}" if msg else "")))
    status = "<div class='status'>" + "".join(f"<div class='box'><b>{E(a)}</b><span>{E(b)}</span></div>" for a, b in boxes) + "</div>"
    warns = []
    if lag is not None and lag > cfg.prev_max_gap:
        y = M.prev_open_day(target, 1, cfg)
        warns.append(f"前日（{y.month}/{y.day}）のデータがまだ取り込めていないため、"
                     "「前日の凹み・据え置き」系の要因は使わずに予想しています。データが届きしだい自動で更新されます。")
    if model.n_days < 120:
        warns.append(f"過去データを集めている途中です（いま{model.n_days}日分／目標 約1年分）。"
                     "自動取得のたびに増えていき、増えるほど店の癖が予想に反映されます。")
    from .specs import spec_for
    unver = sorted({m for m in pred.get("model", []) if not spec_for(m).verified})
    if unver:
        warns.append("スペック未登録の機種があります（仮の数値で推定中）: " + "、".join(unver) + "。hana/specs.py に追記してください。")
    warns = list(extra_warns or []) + warns
    warn_html = "".join(f"<div class='warn'>{E(w)}</div>" for w in warns)

    # 直近の推移
    recent = hist[hist["date"] > hist["date"].max() - pd.Timedelta(days=14)] if len(hist) else hist
    by_unit = {u: g.sort_values("date") for u, g in recent.groupby("unit")} if len(recent) else {}
    last_by_unit = hist.sort_values("date").groupby("unit").tail(1).set_index("unit") if len(hist) else pd.DataFrame()

    trs = []
    for _, r in pred.iterrows():
        u = int(r["unit"])
        p = float(r["score"])
        ratio = p / base if base > 0 else 1
        if isinstance(r.get("why"), list):
            chips = "".join(f"<span class='chip'>{E(t)}</span>" for t in r["why"][:4])
        else:
            chips = "".join(f"<span class='chip {'p' if v > 0 else 'm'}' title='{E(d)}'>{'＋' if v > 0 else '－'}{E(t)}</span>"
                            for t, v, d in M.reasons(r, model))
        lr = last_by_unit.loc[u] if u in last_by_unit.index else None
        prev = "-"
        if lr is not None:
            prev = (f"<div class='prev'>{lr['date']:%m/%d} <b>{fmt_int(lr['games'])}G</b> "
                    f"BB{int(lr['bb'])}/RB{int(lr['rb'])} <b>{fmt_int(lr['diff'], True)}枚</b> "
                    f"{judge_txt(lr)}</div>")
        sp = spark(list(by_unit[u]["judge"].ffill().fillna(3).tail(10))) if u in by_unit else ""
        lo, hi = float(pred["score"].min()), float(pred["score"].max())
        rel = (p - lo) / (hi - lo) * 100 if hi > lo else 50.0       # 今日の台の中での位置（0〜100）
        mark, _ = grade_of(int(r["rank"]), len(pred))
        trs.append(
            f"<tr class='{'top' if r['rank'] <= top_n else ''}'><td class='rank'>{int(r['rank'])}</td>"
            f"<td class='unit'>{u}</td>"
            f"<td class='score'><div class='pct'><span class='mark'>{mark}</span> {rel:.0f}<small>点</small></div>"
            f"<div class='bar'><i style='width:{max(rel, 3):.0f}%'></i></div>"
            + (f"<div class='ratio'>{E(sys_bucket_text(sysinfo, int(r['rank'])))}</div></td>"
               if sysinfo else
               f"<div class='ratio'>{pred_value_text(pred.attrs.get('target'), pred.attrs.get('mth'), r.get('ml_score'))}</div></td>"
               if pred.attrs.get("target") else
               f"<div class='ratio'>同じ条件の台の過去平均：差枚順位 上位{(1 - p) * 100:.0f}%</div></td>"
               if cfg.goal == "rank" else f"<div class='ratio'>見込み {p*100:.1f}%・平常の×{ratio:.2f}</div></td>")
            + f"<td class='reasons'>{chips or '<span class=note>特になし</span>'}</td>"
            f"<td class='prevc'>{prev}</td><td class='sparkc'>{sp}</td></tr>")
    colname = "今日の中での評価"
    spread = (float(pred["score"].max()) - float(pred["score"].min())) * 100 if len(pred) else 0
    meth = {}
    try:
        import json as _json
        meth = _json.loads((Path(pred_dir).parent / "methods.json").read_text(encoding="utf-8")) if pred_dir else {}
    except Exception:
        meth = {}
    if sysinfo:
        rank_note = f"<p class='note'>{E(sys_note(sysinfo, len(pred)))}</p>"
    elif pred.attrs.get("method") and meth.get("ok") and meth.get("kind") == "all":
        v = meth["summary"].get(pred.attrs["method"], {})
        rank_note = (f"<p class='note'>予想の中身5種類×方法5通り（{len(meth['summary'])}通り）を過去{meth['test_days']}日で比べ、"
                     f"一番よく当たった「<b>{E(v.get('target_label', ''))}</b>を<b>{E(v.get('label', ''))}</b>で予想」で並べています"
                     f"（点数は1位＝100点・最下位＝0点）。この方法で2台を比べて上下を当てられた割合は {v.get('acc', 0.5)*100:.1f}%"
                     f"（コイン投げなら50%・偶然でない度合い t={v.get('t', 0):.1f}）。下の「予想方法の比較」に全部の成績があります。</p>")
    elif cfg.goal == "rank":
        rho = (bt or {}).get("rho")
        rank_note = (f"<p class='note'>全{len(pred)}台を、今日の差枚が多くなりそうな順に並べています（点数は1位＝100点・最下位＝0点）。"
                     + (f"過去{bt['days']}日で検証した、予想の順位と実際の差枚順位の一致度は {rho:+.3f}"
                        f"（1＝完全一致・0＝無関係／一致度がプラスだった日 {bt['rho_pos']*100:.0f}%）。" if rho is not None else "")
                     + "</p>")
    else:
        word = "プラス終了の見込み" if cfg.goal == "plus" else "高設定の見込み"
        rank_note = (f"<p class='note'>点数は、今日の{len(pred)}台の中での位置（一番上＝100点・一番下＝0点）です。"
                     f"実際の{word}は1位と最下位で{spread:.1f}ポイントしか違いません"
                     f"（{float(pred['score'].min())*100:.1f}%〜{float(pred['score'].max())*100:.1f}%）。"
                     "過去データで差が小さいと出ているためで、無理に差を広げると検証の成績が悪くなりました。"
                     "順位の参考としてご覧ください。</p>")
    table = rank_note + (f"<table class='rk'><tr class='hd'><th>順位</th><th>台番</th><th>{colname}</th><th>主な根拠</th><th>前回データ</th>"
             "<th>直近の判定設定</th></tr>" + "".join(trs) + "</table>")
    judgesec = judge_grid(hist, pred["unit"].astype(int).tolist(), cfg)
    daysec = day_section(hist, target, cfg)

    # 島マップ
    units_sorted = sorted(pred["unit"].astype(int).tolist())
    sc = dict(zip(pred["unit"].astype(int), pred["score"]))
    rk = dict(zip(pred["unit"].astype(int), pred["rank"].astype(int)))
    runs: list[list[int]] = []
    if cfg.islands:
        runs = [[u for u in units_sorted if a <= u <= b] for a, b in cfg.islands]
    else:
        cur: list[int] = []
        for u in units_sorted:
            if cur and u != cur[-1] + 1:
                runs.append(cur)
                cur = []
            cur.append(u)
        if cur:
            runs.append(cur)
    isl = "".join("<div class='island'>" + "".join(
        f"<div class='cell' style='background:{grade_of(rk[u], len(pred))[1]}'>{u}<small>{grade_of(rk[u], len(pred))[0]}</small></div>"
        for u in run) + "</div>" for run in runs if run)
    mapsec = f"<h2>島マップ</h2><div class='map'>{isl}</div><p class='note'>色が濃いほど今日の中で評価が高い台（◎上位10%・○上位30%・△中位・×下位30%）。</p>"

    # 店の癖
    s = model.summaries
    habit = ""
    if s:
        parts = [habit_table(s.get("seq2"), base, "判定の流れ（2日前→前日）→ 当日"),
                 habit_table(s.get("seq3"), base, "判定の流れ（3日前→2日前→前日）→ 当日"),
                 habit_table(s.get("digit"), base, "日付の末尾別（ハナハナ）"),
                 habit_table(s.get("wday"), base, "曜日別（ハナハナ）"),
                 habit_table(s.get("match"), base, "台番末尾と日付末尾の一致"),
                 habit_table(s.get("prevdiff"), base, "前日の差枚 → 当日"),
                 habit_table(s.get("prevhigh"), base, "前日の挙動 → 当日（据え置き傾向）"),
                 habit_table(s.get("corner"), base, "台の位置"),
                 habit_table(s.get("special"), base, "特定日")]
        ut = s.get("unit")
        if ut is not None and len(ut):
            ut = ut.sort_values("rate", ascending=False)
            parts.append(habit_table(pd.concat([ut.head(6), ut.tail(4)]), base, "台番別（入りやすい台・入りにくい台）"))
        hd = s.get("hall_digit")
        if hd is not None and len(hd):
            span = s.get("hall_span")
            rows = "".join(f"<tr><td>{int(r['digit'])}のつく日</td><td class='num'>{fmt_int(r['mean'], True)}</td>"
                           f"<td class='num'>{int(r['count'])}日</td></tr>" for _, r in hd.iterrows())
            parts.append(f"<div><h3 style='font-size:15px;margin:6px 0'>店全体（全機種）の平均差枚・日付末尾別</h3>"
                         f"<table class='mini'><tr><th>日付</th><th>平均差枚/台</th><th>日数</th></tr>{rows}</table>"
                         f"<p class='note'>{span[0]}〜{span[1]}（{span[2]}日）</p></div>")
        rate_def = (f"率 = {cfg.label_min_games:,}G以上回った台の、その日の差枚順位の平均（上位＝100%・下位＝0%）"
                    if cfg.goal == "rank" else
                    f"率 = {cfg.label_min_games:,}G以上回った台のうち、差枚プラスで終わった台の割合"
                    if cfg.goal == "plus" else
                    f"率 = その日の合算で「設定{cfg.high_min}以上」と判定された台の割合")
        habit = (f"<h2>学習した店の癖</h2><p class='note'>{rate_def}"
                 f"。全体平均は {base*100:.1f}%。日数が少ない条件は偶然の可能性があります。</p>"
                 f"<div class='grid2'>{''.join(p for p in parts if p)}</div>")

    # 予想方法の比較
    msec = ""
    if meth.get("ok") and meth.get("kind") == "all":
        rows_m = "".join(
            f"<tr{' style=\'font-weight:700\'' if k == meth.get('best') else ''}><td>{E(v['target_label'])}</td>"
            f"<td>{E(v['label'])}{' ★採用' if k == meth.get('best') else ''}</td>"
            f"<td class='num'>{v['acc']*100:.1f}%</td><td class='num'>{v['t']:.1f}</td>"
            f"<td class='num'>{v['first']*100:.1f}%／{v['second']*100:.1f}%</td></tr>"
            for k, v in sorted(meth["summary"].items(), key=lambda kv: -kv[1]["t"]))
        msec = (f"<h2>予想方法の比較</h2><p class='note'>過去{meth['test_days']}日の各日を、その日より前のデータだけで予想し直して採点"
                f"（{meth.get('when', '')}時点・週1回とデータが2割増えるたびにやり直し）。当たる割合＝その日の台を2台ずつ比べて、"
                "予想で上にした方が実際にも上だった割合（プラス終了はAUC）。50%がコイン投げ。t＝偶然でない度合い（2以上でおおむね偶然ではない）。"
                "前半・後半の両方で50%を上回ったものの中から、tが一番大きいものを採用します。</p>"
                f"<div class='scroll'><table class='mini'><tr><th>予想する中身</th><th>方法</th><th>当たる割合</th><th>t</th>"
                f"<th>前半／後半</th></tr>{rows_m}</table></div>")
    elif meth.get("ok"):
        rows_m = "".join(
            f"<tr{' style=\'font-weight:700\'' if k == meth['best'] else ''}><td>{E(v['label'])}{' ★採用' if k == meth['best'] else ''}</td>"
            f"<td class='num'>{v['rho']:+.3f}</td><td class='num'>{v['rho_first']:+.3f}／{v['rho_second']:+.3f}</td>"
            f"<td class='num'>{v['top5_diff']:+,.0f}</td><td class='num'>{v['top5_plus']*100:.0f}%</td></tr>"
            for k, v in sorted(meth["summary"].items(), key=lambda kv: -kv[1]["rho"]))
        anyv = next(iter(meth["summary"].values()))
        msec = (f"<h2>予想方法の比較</h2><p class='note'>過去{meth['test_days']}日の各日を、その日より前のデータだけで予想し直して採点"
                f"（{meth.get('when', '')}時点・週1回とデータが2割増えるたびにやり直し）。一致度＝全台の予想順位と実際の差枚順位の順位相関"
                f"（1＝完全一致・0＝無関係）。前半と後半の両方でプラスだった方法の中から、一番高いものを採用します。"
                f"全台平均は 差枚{anyv['all_diff']:+,.0f}枚・プラス率{anyv['all_plus']*100:.0f}%。</p>"
                f"<div class='scroll'><table class='mini'><tr><th>方法</th><th>一致度</th><th>前半／後半</th><th>上位5台の平均差枚</th>"
                f"<th>上位5台のプラス率</th></tr>{rows_m}</table></div>")

    if sysinfo:      # 予想システムのときは、店の癖・答え合わせ・方法の比較を新しい形で出す
        habit = sys_habits_section(sysinfo)
        msec = ""
    # 検証
    btsec = msec + "<h2>過去データでの検証</h2>"
    if bt and bt.get("ok"):
        btsec += (f"<div class='status'>"
                  + (f"<div class='box'><b>{bt['rho']:+.3f}</b><span>全台の予想順位と実際の差枚順位の一致度（順位相関・1が完全一致）"
                     f"／プラスだった日 {bt['rho_pos']*100:.0f}%</span></div>"
                     if cfg.goal == "rank" and "rho" in bt else
                     f"<div class='box'><b>{bt['top_y']*100:.1f}%</b><span>予想上位{bt['top_k']}台のプラス終了率"
                     f"（全台 {bt['all_y']*100:.1f}%）</span></div>"
                     if cfg.goal == "plus" and "top_y" in bt else
                     f"<div class='box'><b>{bt['hit']*100:.1f}%</b><span>上位{bt['top_k']}台予想が、その日の推定上位{bt['top_k']}台に入った率"
                     f"（ランダムなら{bt['rand_hit']*100:.1f}%）</span></div>") +
                  f"<div class='box'><b>{bt['top_diff']:+,.0f}枚</b><span>予想上位{bt['top_k']}台の平均差枚（全台平均 {bt['all_diff']:+,.0f}枚）</span></div>"
                  f"<div class='box'><b>{bt['win_days']*100:.0f}%</b><span>予想上位が全台平均を上回った日の割合</span></div>"
                  f"<div class='box'><b>{bt['days']}日</b><span>検証日数（データ公開の遅れ {bt['lag']}日を想定）</span></div></div>"
                  f"<p class='note'>各日について、その日より前のデータだけで予想し直して答え合わせした結果です（{bt.get('when', '')}時点）。</p>")
    else:
        reason = bt.get("reason") if bt else "まだ実行していません"
        btsec += f"<p class='note'>{E(reason)}</p>"

    if sysinfo:
        btsec = sys_answer_section(sysinfo) + sys_research_section(sysinfo)
    cov_txt = (f"データ: {cov.get('unit_first')}〜{cov.get('unit_last')}（{cov.get('unit_days')}日・{cov.get('unit_rows'):,}台日）"
               if cov.get("unit_rows") else "データなし")
    from datetime import timedelta, timezone
    now_jst = datetime.now(timezone(timedelta(hours=9)))
    foot = (f"<footer>{E(cov_txt)}／取得元: アナスロ・みんレポ（公開データ）／作成 {now_jst:%Y-%m-%d %H:%M}<br>"
            + (f"{E(info_line)}<br>" if info_line else "") +
            "この予想は公開データから計算した確率であり、結果を保証するものではありません。遊技は余裕のある範囲でお楽しみください。</footer>")

    doc = (f"<!doctype html><html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>ハナハナ予想 {target:%m/%d}</title><style>{CSS}</style></head><body><div class='wrap'>"
           f"{head}{status}{warn_html}{daysec}<h2>台別ランキング</h2>"
           + ("" if sysinfo else f"<p class='note'>{E(factor_note(model))}</p>")
           + f"{table}{mapsec}{quit_section((sysinfo or {}).get('quit'))}{judgesec}{check_section(hist, pred_dir, target)}"
           f"{habit}{btsec}{foot}</div></body></html>")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{target:%Y-%m-%d}.html"
    path.write_text(doc, encoding="utf-8")
    (out_dir / "latest.html").write_text(doc, encoding="utf-8")
    (out_dir / "index.html").write_text(doc, encoding="utf-8")
    cols = ["rank", "unit", "model", "score"]
    pred.assign(reasons=[" / ".join(t for t, _, _ in M.reasons(r, model)) for _, r in pred.iterrows()])[
        cols + ["reasons"]].to_csv(out_dir / f"{target:%Y-%m-%d}.csv", index=False, encoding="utf-8-sig")
    return path
