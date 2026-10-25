"""Regressions for contradictions observed in delivered HK email reports."""

from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from data_provider.realtime_types import RealtimeSource, UnifiedRealtimeQuote
from src.analyzer import AnalysisResult, GeminiAnalyzer, stabilize_decision_with_structure
from src.core.pipeline import StockAnalysisPipeline
from src.notification import NotificationService
from src.notification_sender.email_sender import EmailSender
from src.config import Config
from src.reports.news_sources import format_news_sources
from src.reports.report_time import build_report_time, normalize_next_session_dates
from src.reports.report_time import format_report_time
from src.report_language import get_report_labels
from src.search_service import SearchResponse, SearchResult, SearchService
from src.stock_analyzer import TrendAnalysisResult


def _pipeline():
    pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
    pipeline.config = SimpleNamespace(enable_realtime_technical_indicators=True, report_language="zh")
    pipeline.search_service = SimpleNamespace(news_window_days=3)
    return pipeline


def _context():
    return {"code": "hk01810", "date": "2026-10-23", "today": {
        "date": "2026-10-23", "open": 25.8, "high": 27.1, "low": 25.5, "close": 26.8,
        "ma5": 26.5, "ma10": 26.0, "ma20": 25.0,
    }, "yesterday": {"close": 25.8}}


def _quote(**overrides):
    values = dict(code="hk01810", source=RealtimeSource.TENCENT, price=26.8,
                  open_price=25.8, high=27.1, low=25.5)
    values.update(overrides)
    return UnifiedRealtimeQuote(**values)


@pytest.mark.parametrize("overrides", [
    {"open_price": None, "high": None, "low": None, "pre_close": 25.8},
    {"high": 26.8, "low": 26.8},
    {"high": float("nan")},
    {"is_stale": True},
    {"provider_timestamp": "2026-10-22T15:00:00+08:00"},
])
def test_incomplete_or_invalid_quotes_preserve_daily_bar_and_snapshot(overrides):
    pipeline, context = _pipeline(), _context()
    original = deepcopy(context)
    quote = _quote(**overrides)
    now = datetime(2026, 10, 23, 15, tzinfo=ZoneInfo("Asia/Hong_Kong"))
    with patch("src.core.pipeline.get_market_now", side_effect=lambda market, current_time=None: current_time or now):
        enhanced = pipeline._enhance_context(context, quote, None, TrendAnalysisResult(code="hk01810", ma5=26.5),
                                             fundamental_context={})
        frame = pd.DataFrame([context["today"]])
        assert pipeline._augment_historical_with_realtime(frame, quote, "hk01810") is frame
    assert context == original
    assert enhanced["today"] == original["today"]
    assert enhanced["date"] == "2026-10-23"
    assert enhanced["realtime"]["price"] == 26.8
    snapshot = GeminiAnalyzer.__new__(GeminiAnalyzer)._build_market_snapshot(enhanced)
    assert snapshot["low"] == "25.50"
    assert snapshot["high"] == "27.10"
    assert snapshot["amplitude"] != "0.00%"


def test_weekend_and_disabled_overlay_preserve_completed_bar():
    pipeline, context = _pipeline(), _context()
    for now, enabled in [(datetime(2026, 10, 24, 10), True), (datetime(2026, 10, 23, 15), False)]:
        pipeline.config.enable_realtime_technical_indicators = enabled
        with patch("src.core.pipeline.get_market_now", return_value=now):
            enhanced = pipeline._enhance_context(context, _quote(price=27.0), None,
                                                 TrendAnalysisResult(code="hk01810", ma5=26.5),
                                                 fundamental_context={})
        assert enhanced["today"] == context["today"]
        assert enhanced["date"] == "2026-10-23"


def test_guarded_buy_has_one_consistent_plan_in_email_and_template():
    result = AnalysisResult(code="hk01810", name="小米集团", sentiment_score=70,
                            trend_prediction="看多", operation_advice="买入", decision_type="buy",
                            current_price=26.8, dashboard={"battle_plan": {
                                "sniper_points": {"ideal_buy": "26.62-26.80 首批买入", "stop_loss": "25.50"},
                                "position_strategy": {"suggested_position": "4-5成", "entry_plan": "现价先买2成"},
                                "action_checklist": ["现价买入2成"],
                            }})
    stabilize_decision_with_structure(result, None, {"capital_flow": {"status": "failed", "data": {}}})
    assert result.decision_type == "hold"
    assert result.dashboard["battle_plan"]["sniper_points"]["stop_loss"] == "25.50"
    assert "恢复" not in result.get_position_advice()
    service = NotificationService()
    for renderer in (False, True):
        with patch.object(service._config, "report_renderer_enabled", renderer):
            report = service.generate_dashboard_report([result])
        for stale_plan in ("4-5成", "现价先买2成", "首批买入", "现价买入2成"):
            assert stale_plan not in report
        assert "空仓暂不建仓" in report


def test_all_irrelevant_news_is_removed_before_model_prompt():
    service = SearchService(searxng_public_instances_enabled=False)
    response = SearchResponse(query="MiniMax news", provider="fixture", success=True, results=[
        SearchResult(title="NFL scores", snippet="Football results", url="https://example.com/nfl", source="fixture"),
        SearchResult(title="New York POWER Act", snippet="Local legislative update", url="https://example.com/power", source="fixture"),
    ])
    ranked = service._rank_news_response(response, stock_code="hk00100", stock_name="MiniMax",
                                         prefer_chinese=False, max_results=6, log_scope="regression")
    admitted = service._filter_ranked_news_for_context(ranked, log_scope="regression")
    assert admitted.results == []
    prompt = service.format_intel_report({"latest_news": admitted}, "MiniMax")
    assert "NFL scores" not in prompt
    assert "POWER Act" not in prompt
    assert "未找到相关信息" in prompt


def test_news_links_survive_prompt_and_both_report_renderers():
    service = SearchService(searxng_public_instances_enabled=False)
    item = SearchResult(title="腾讯控股回购公告", snippet="腾讯控股公布回购", url="https://www.hkexnews.hk/example.pdf",
                        source="HKEX", published_date="2026-10-23", relevance_score=100,
                        relevance_category="direct_company_news")
    context = service.format_intel_report({"latest_news": SearchResponse(query="腾讯", results=[item],
                                         provider="fixture", success=True)}, "腾讯")
    assert item.url in context
    result = AnalysisResult(code="hk00700", name="腾讯控股", sentiment_score=50,
                            trend_prediction="震荡", operation_advice="观望", dashboard={
                                "news_sources": [{"title": item.title, "url": item.url, "source": item.source,
                                                  "published_date": item.published_date, "category": item.relevance_category}],
                                "data_perspective": {"volume_analysis": {"volume_ratio": None, "turnover_rate": None}},
                            })
    notifier = NotificationService()
    for renderer in (False, True):
        with patch.object(notifier._config, "report_renderer_enabled", renderer):
            report = notifier.generate_dashboard_report([result])
        assert item.url in report
        assert item.published_date in report
        assert "None" not in report
    assert "javascript:" not in format_news_sources([{"title": "bad", "url": "javascript:alert(1)"}])


def test_hk_calendar_dates_and_event_dates_remain_distinct():
    now = datetime(2026, 10, 25, 0, 32, tzinfo=ZoneInfo("Asia/Hong_Kong"))
    with patch("src.reports.report_time.get_market_now", return_value=now):
        timing = build_report_time("hk01810", "2026-10-23")
    assert timing["generated_at"] == "2026-10-25T00:32:00+08:00"
    assert timing["daily_bar_date"] == "2026-10-23"
    assert timing["next_trading_date"] == "2026-10-26"
    result = AnalysisResult(code="hk01810", name="小米", sentiment_score=50, trend_prediction="震荡",
                            operation_advice="观望", dashboard={
                                "phase_decision": {"next_check_time": "下一交易日（2026-10-27）开盘"},
                                "intelligence": {"latest_news": "2026-10-26 发布产品"},
                            })
    normalize_next_session_dates(result, timing)
    assert "2026-10-26" in result.dashboard["phase_decision"]["next_check_time"]
    assert "2026-10-26" in result.dashboard["intelligence"]["latest_news"]


def test_comprehensive_news_deduplicates_before_counting_and_prompting():
    service = SearchService(searxng_public_instances_enabled=False)
    item = SearchResult(title="腾讯控股 00700 发布回购公告", snippet="股份回购公告",
                        url="https://www.hkexnews.hk/one-source.pdf", source="HKEX",
                        published_date=datetime.now().date().isoformat())
    service._providers = [SimpleNamespace(is_available=True, name="fixture", search=lambda *args, **kwargs:
        SearchResponse(query="腾讯", provider="fixture", success=True, results=[deepcopy(item)]))]
    with patch("src.search_service.time.sleep"):
        intel = service.search_comprehensive_intel("hk00700", "腾讯控股", max_searches=2)
    assert sum(len(response.results) for response in intel.values()) == 1
    assert service.format_intel_report(intel, "腾讯").count(item.url) == 1


def test_hk_email_subject_uses_same_local_date_as_report():
    config = Config(stock_list=["hk01810"], email_sender="sender@qq.com", email_password="test-only",
                    email_receivers=["receiver@example.com"])
    sender = EmailSender(config)
    now = datetime(2026, 10, 25, 0, 32, tzinfo=ZoneInfo("Asia/Hong_Kong"))
    with patch("src.core.trading_calendar.get_market_now", return_value=now), \
            patch("src.notification_sender.email_sender.smtplib.SMTP_SSL") as smtp:
        assert sender.send_to_email("# 2026-10-25 决策仪表盘")
    message = smtp.return_value.send_message.call_args.args[0]
    assert "2026-10-25" in str(message["Subject"])


@pytest.mark.parametrize("language", ["zh", "en", "ko"])
@pytest.mark.parametrize("renderer", [False, True])
def test_position_scope_is_explicit_across_notification_formats(language, renderer):
    result = AnalysisResult(code="hk01810", name="小米", sentiment_score=50, trend_prediction="震荡",
                            operation_advice="观望", report_language=language)
    notifier = NotificationService()
    with patch.object(notifier._config, "report_renderer_enabled", renderer):
        reports = [notifier.generate_dashboard_report([result]), notifier.generate_brief_report([result]),
                   notifier.generate_wechat_dashboard([result]), notifier.generate_single_stock_report(result)]
    note = get_report_labels(language)["position_scope_note"]
    assert all(note in report for report in reports)


def test_next_check_keeps_specific_future_events_after_market_close():
    result = AnalysisResult(code="hk00700", name="腾讯", sentiment_score=50, trend_prediction="震荡",
        operation_advice="观望", dashboard={"phase_decision": {
            "next_check_time": "2026-11-03 财报公布后检查；上一数据日 2026-10-23"
        }})
    result.market_phase_summary = {"phase": "postmarket"}
    normalize_next_session_dates(result, {"next_trading_date": "2026-10-26"})
    assert result.dashboard["phase_decision"]["next_check_time"] == (
        "2026-11-03 财报公布后检查；上一数据日 2026-10-23"
    )


def test_unknown_calendar_does_not_allow_daily_bar_overlay():
    with patch("src.core.trading_calendar._XCALS_AVAILABLE", False):
        assert not _pipeline()._can_overlay_realtime_bar(_quote(), "hk01810")


def test_hk_market_news_uses_topic_search_and_keeps_macro_material():
    from src.market_analyzer import MarketAnalyzer
    from src.core.market_profile import HK_PROFILE

    service = SearchService(searxng_public_instances_enabled=False)
    item = SearchResult(title="Hong Kong stocks rally as investors assess interest rates",
        snippet="Hang Seng Index rises as global markets digest central bank policy.",
        url="https://example.com/hongkong", source="fixture", published_date=datetime.now().date().isoformat())
    service._providers = [SimpleNamespace(is_available=True, name="fixture", search=lambda *args, **kwargs:
        SearchResponse(query="Hong Kong market", provider="fixture", success=True, results=[deepcopy(item)]))]
    analyzer = MarketAnalyzer.__new__(MarketAnalyzer)
    analyzer.region, analyzer.profile = "hk", HK_PROFILE
    analyzer.config = SimpleNamespace(report_language="zh")
    analyzer.search_service = service
    result = analyzer.search_market_news()
    assert [news.url for news in result] == [item.url]


def test_index_actual_session_date_survives_provider_and_market_overview():
    from data_provider.yfinance_fetcher import YfinanceFetcher
    from src.market_analyzer import MarketAnalyzer
    from src.core.market_profile import HK_PROFILE

    hist = pd.DataFrame({"Open": [99., 100.], "High": [102., 103.], "Low": [98., 99.],
                         "Close": [100., 101.], "Volume": [1000, 1100]},
                        index=pd.DatetimeIndex(["2026-10-22", "2026-10-23"]))
    yf = SimpleNamespace(Ticker=lambda symbol: SimpleNamespace(history=lambda **kwargs: hist))
    item = YfinanceFetcher()._fetch_yf_ticker_data(yf, "^HSI", "恒生指数", "HSI")
    analyzer = MarketAnalyzer.__new__(MarketAnalyzer)
    analyzer.region, analyzer.profile = "hk", HK_PROFILE
    analyzer.config = SimpleNamespace(report_language="zh")
    analyzer.data_manager = SimpleNamespace(get_main_indices=lambda region: [item])
    with patch("src.market_analyzer.get_effective_trading_date", return_value=datetime(2026, 10, 22).date()):
        overview = analyzer.get_market_overview()
    assert overview.date == "2026-10-23"
    assert overview.indices[0].to_dict()["date"] == "2026-10-23"
    assert "数据源行情日期: 恒生指数: 2026-10-23" in analyzer._build_indices_block(overview)


def test_report_discloses_daily_bars_behind_expected_completed_session():
    now = datetime(2026, 10, 25, 0, 32, tzinfo=ZoneInfo("Asia/Hong_Kong"))
    with patch("src.reports.report_time.get_market_now", return_value=now):
        timing = build_report_time("hk00700", "2026-10-22")
    assert timing["expected_daily_bar_date"] == "2026-10-23"
    assert timing["daily_bar_stale"] is True
    assert "日线更新滞后" in format_report_time(timing)
