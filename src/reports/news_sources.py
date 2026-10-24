"""Render retrieved news references without relying on model-generated URLs."""

from html import escape
from typing import Any
from urllib.parse import quote, urlsplit


def format_news_sources(sources: Any, language: str = "zh") -> str:
    if not isinstance(sources, list):
        return ""
    english = language in {"en", "ko"}
    rows = []
    seen = set()
    for item in sources:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        try:
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                continue
        except ValueError:
            continue
        if url in seen:
            continue
        seen.add(url)
        title = escape(" ".join(str(item.get("title") or url).split()))
        title = title.replace("[", "\\[").replace("]", "\\]")
        category = item.get("category")
        scope = ("Company-related" if english else "公司相关") if category == "direct_company_news" else (
            "Sector / macro context" if english else "行业 / 宏观背景"
        )
        meta = " / ".join(escape(str(value)) for value in (
            item.get("source"), item.get("published_date") or ("Date unavailable" if english else "日期未提供"), scope
        ) if value)
        rows.append(f"- [{title}]({quote(url, safe=':/?&=%#@+;,~')})（{meta}）")
    if not rows:
        return ""
    heading = "Retrieved news sources" if english else "新闻检索来源"
    return f"**{heading}**\n\n" + "\n".join(rows)
