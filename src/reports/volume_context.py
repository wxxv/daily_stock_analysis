"""Describe computed daily volume ratios separately from realtime indicators."""

import math
from typing import Any


def ground_volume_analysis(result: Any) -> None:
    """Keep persisted model fields consistent with an authoritative quote snapshot."""
    snapshot = getattr(result, "market_snapshot", None)
    dashboard = getattr(result, "dashboard", None)
    if not isinstance(snapshot, dict) or not isinstance(dashboard, dict):
        return
    perspective = dashboard.get("data_perspective")
    volume = perspective.get("volume_analysis") if isinstance(perspective, dict) else None
    if not isinstance(volume, dict):
        return
    for field in ("volume_ratio", "turnover_rate"):
        if field not in snapshot:
            continue
        try:
            value = float(str(snapshot[field]).strip().rstrip("%"))
            volume[field] = value if math.isfinite(value) else None
        except (TypeError, ValueError):
            volume[field] = None


def format_volume_analysis(snapshot: Any, analysis: Any, language: str = "zh") -> str:
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    analysis = analysis if isinstance(analysis, dict) else {}

    def numeric(field: str, percent: bool = False) -> str:
        value = snapshot[field] if field in snapshot else analysis.get(field)
        if isinstance(value, bool):
            return "N/A"
        try:
            number = float(str(value).strip().rstrip("%"))
        except (TypeError, ValueError):
            return "N/A"
        if not math.isfinite(number):
            return "N/A"
        return f"{number:.2f}" + ("%" if percent else "")

    status = analysis.get("volume_status") or "N/A"
    if language in {"en", "ko"}:
        return f"**Volume status**: {status} | Realtime volume ratio: {numeric('volume_ratio')} | Turnover: {numeric('turnover_rate', True)}"
    return f"**日线量能状态**: {status} | 实时量比: {numeric('volume_ratio')} | 换手率: {numeric('turnover_rate', True)}"


def format_volume_reference(snapshot: Any, language: str = "zh") -> str:
    if not isinstance(snapshot, dict) or not snapshot:
        return ""
    english = language in {"en", "ko"}
    parts = []
    for field, zh, en in (("daily_volume_vs_previous", "日成交量/前日成交量", "Daily volume / previous-session volume"),
                          ("daily_volume_vs_5d", "日成交量/前5日均量", "Daily volume / prior-5-session average")):
        value = snapshot.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            parts.append(f"{en if english else zh}: {value:.2f}×")
    note = ("Realtime volume ratio and turnover are separate metrics; N/A means the source did not provide the value."
            if english else "实时量比、换手率为独立指标；N/A 表示数据源未提供，不能用日线成交量倍数替代。")
    return " | ".join(parts + [note])
