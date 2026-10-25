"""Disclose failed analyses without exposing raw provider exceptions."""

from typing import Any


def with_analysis_failures(report: str, failures: Any, language: str = "zh") -> str:
    if not isinstance(failures, dict) or not failures:
        return report
    english = language in {"en", "ko"}
    title = "Analysis incomplete" if english else "本轮分析未全部完成"
    reason = "No valid report was generated; excluded from scoring and recommendations." if english else "未生成有效报告，未计入评分和买卖建议。"
    rows = [f"> **{title}**: {', '.join(str(code) for code in failures)}. {reason}"]
    for code, category in failures.items():
        if english:
            detail = "Model returned empty content after retry" if category == "empty_response" else "Analysis failed; see run logs"
        else:
            detail = "模型返回空内容，重试后仍失败" if category == "empty_response" else "分析失败，详见运行日志"
        rows.append(f"> - {code}: {detail}")
    return "\n".join(rows) + "\n\n" + report
