"""Market-local report timestamps and calendar-backed planning dates."""

import re
from datetime import date
from typing import Any, Dict, Optional

from src.core.trading_calendar import get_effective_trading_date, get_market_for_stock, get_market_now, get_next_trading_date


def daily_bar_is_stale(actual_date: Any, expected_date: Any) -> Optional[bool]:
    """Compare known session dates; missing calendars/dates remain unknown."""
    if not actual_date or not expected_date:
        return None
    try:
        return date.fromisoformat(str(actual_date)[:10]) < date.fromisoformat(str(expected_date)[:10])
    except ValueError:
        return None


def build_report_time(code: str, daily_bar_date: Any) -> Dict[str, Any]:
    market = get_market_for_stock(code)
    now = get_market_now(market)
    next_session = get_next_trading_date(market, now.date())
    expected_date = get_effective_trading_date(market, current_time=now).isoformat() if next_session else None
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "daily_bar_date": str(daily_bar_date) if daily_bar_date else None,
        "next_trading_date": next_session.isoformat() if next_session else None,
        "expected_daily_bar_date": expected_date,
        "daily_bar_stale": daily_bar_is_stale(daily_bar_date, expected_date),
    }


def format_report_time(timing: Optional[Dict[str, Any]], language: str = "zh") -> str:
    if not isinstance(timing, dict) or not timing:
        return ""
    english = language in {"en", "ko"}
    missing = "Unavailable" if english else "待确认"
    generated = timing.get("generated_at") or missing
    daily = timing.get("daily_bar_date") or missing
    next_date = timing.get("next_trading_date") or missing
    if english:
        text = f"Generated: {generated} | Daily bar as of: {daily} | Next trading session: {next_date}"
        if timing.get("daily_bar_stale") is True:
            text += f" | Daily bars are stale; expected through {timing.get('expected_daily_bar_date')}"
        return text
    text = f"生成时间：{generated} | 日线截至：{daily} | 下一交易日：{next_date}"
    if timing.get("daily_bar_stale") is True:
        text += f" | 日线更新滞后，应至少更新至 {timing.get('expected_daily_bar_date')}"
    return text


_NEXT_SESSION_DATE = re.compile(
    r"((?:下一(?:个)?交易日|next trading (?:day|session))\s*[:：]?\s*[（(]?\s*)\d{4}-\d{2}-\d{2}",
    re.IGNORECASE,
)


def normalize_next_session_dates(result: Any, timing: Dict[str, Any]) -> None:
    """Correct explicit next-session labels without touching dated news/events."""
    language = getattr(result, "report_language", "zh")
    next_date = timing.get("next_trading_date") or ("date unconfirmed" if language in {"en", "ko"} else "日期待确认")

    def normalize(value: Any) -> Any:
        if isinstance(value, str):
            return _NEXT_SESSION_DATE.sub(lambda match: match.group(1) + next_date, value)
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {key: (item if key in {"news_sources", "intelligence", "report_time"} else normalize(item))
                    for key, item in value.items()}
        return value

    result.dashboard = normalize(result.dashboard or {})
    result.dashboard["report_time"] = dict(timing)
    for field in ("analysis_summary", "short_term_outlook", "medium_term_outlook", "buy_reason", "risk_warning"):
        setattr(result, field, normalize(getattr(result, field, "")))
