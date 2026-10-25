"""Keep statement currencies and per-field periods visible to models and readers."""

from typing import Any, Dict, Optional


def format_financial_basis(
    report: Dict[str, Any], dividend: Optional[Dict[str, Any]] = None,
    language: str = "zh", *, for_prompt: bool = False,
) -> str:
    if not isinstance(report, dict) or not report:
        return ""
    english = language in {"en", "ko"}
    unknown = "unconfirmed" if english else "待确认"
    currency = report.get("currency") or unknown
    dividend_currency = (dividend.get("currency") if isinstance(dividend, dict) else None) or unknown
    if english:
        text = f"Financial statement currency: {currency}; dividend currency: {dividend_currency}."
        if for_prompt:
            text += (" Absolute financial values are in base currency units, not millions. "
                     "Always state the currency in prose; never infer it from the listing market or convert without an exchange rate. "
                     "ROE and growth values use percent units (12.5 means 12.5%).")
        names = {"revenue": "Revenue", "net_profit_parent": "Net profit", "operating_cash_flow": "Operating cash flow"}
        bases = {"statement": "source statement (quarter/cumulative basis unverified)",
                 "ttm": "TTM", "estimated_ttm": "estimated TTM (revenue × profit margin)"}
    else:
        text = f"财报币种：{currency}；分红币种：{dividend_currency}。"
        if for_prompt:
            text += ("财报金额原值以基础货币单位计，非万元或百万元。"
                     "正文金额必须注明币种，不得按上市市场或股价币种推算，也不得无汇率擅自换算。"
                     "ROE及同比为百分数值，例如12.5表示12.5%。")
        names = {"revenue": "营业收入", "net_profit_parent": "归母净利润", "operating_cash_flow": "经营现金流"}
        bases = {"statement": "数据源报表值（单季/累计口径未核验）", "ttm": "TTM近12个月",
                 "estimated_ttm": "TTM估算（营收×净利率，非直接报表值）"}
    periods = report.get("field_periods") if isinstance(report.get("field_periods"), dict) else {}
    for key, name in names.items():
        if report.get(key) is None:
            continue
        period = periods.get(key) if isinstance(periods.get(key), dict) else {}
        basis = bases.get(period.get("basis"), unknown)
        end = period.get("period_end") or (report.get("report_date") if not period else None) or unknown
        text += f" {name}: {basis}, {end}." if english else f" {name}：{basis}，截至 {end}。"
    text += (" Missing dates/values are unknown, not zero. Do not compare unlike periods."
             if english else " 缺少期间或数值表示未知，不等于零；不得将不同期间直接比较。")
    return text
