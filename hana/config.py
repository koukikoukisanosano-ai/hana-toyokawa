from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path | None = None) -> dict:
    p = path or ROOT / "config.yaml"
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("hall", {})
    cfg.setdefault("target", {})
    cfg.setdefault("fetch", {})
    cfg.setdefault("predict", {})
    return cfg


def model_settings(cfg: dict):
    from .model import Settings
    t, p, h = cfg["target"], cfg["predict"], cfg["hall"]
    islands = t.get("islands") or None
    return Settings(
        keyword=t.get("keyword", "ハナハナ"),
        high_min=int(t.get("high_min", 5)),
        judge_min_games=int(t.get("judge_min_games", 2000)),
        label_min_games=int(t.get("train_min_games", 1000)),
        goal=str(t.get("goal", "plus")),
        event_digits=tuple(h.get("event_digits", [0, 5])),
        prev_max_gap=int(p.get("prev_max_gap", 1)),
        islands=[list(map(int, x)) for x in islands] if islands else None,
        closed_days=closed_days(cfg),
    )


def closed_days(cfg: dict) -> tuple[str, ...]:
    """休業日: config.yaml の closed_days ＋ 自動で見つけた休業日（data/closed_days.txt）。"""
    days = {str(d) for d in (cfg["hall"].get("closed_days") or [])}
    p = ROOT / "data" / "closed_days.txt"
    if p.exists():
        days |= {x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()}
    return tuple(sorted(days))
