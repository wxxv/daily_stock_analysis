"""Regression paths observed in the delivered HK report, through real consumers."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_provider.yfinance_fetcher import YfinanceFetcher
from data_provider.yfinance_fundamental_adapter import YfinanceFundamentalAdapter
from src.analyzer import AnalysisResult, GeminiAnalyzer, _AllModelsFailedError, stabilize_decision_with_structure
from src.config import Config
from src.core.market_profile import HK_PROFILE
from src.enums import ReportType
from src.market_analyzer import MarketAnalyzer, MarketIndex, MarketOverview
from src.notification import NotificationService
from src.reports.volume_context import ground_volume_analysis
from src.share_image import build_share_image_html
from tests import test_pipeline_single_stock_notify as batch_fixture
from tests import test_pipeline_notification_image_routing as email_fixture
from tests.test_pipeline_single_stock_notify import _make_result
from src.core.market_strategy import get_market_strategy_blueprint


def _market():
    analyzer = MarketAnalyzer.__new__(MarketAnalyzer)
    analyzer.region, analyzer.profile = "hk", HK_PROFILE
    analyzer.config = SimpleNamespace(report_language="zh")
    analyzer.strategy = get_market_strategy_blueprint("hk")
    return analyzer


@pytest.mark.parametrize("previous", [None, 0., float("nan"), float("inf")])
def test_unknown_previous_close_survives_provider_prompt_template_and_market_score(previous):
    hist = pd.DataFrame({"Open": [100.], "High": [103.], "Low": [99.], "Close": [101.], "Volume": [1000.]},
                        index=pd.DatetimeIndex(["2026-10-23"]))
    if previous is not None:
        prior = hist.copy()
        prior.index = pd.DatetimeIndex(["2026-10-22"])
        prior["Close"] = previous
        hist = pd.concat([prior, hist])
    yf = SimpleNamespace(Ticker=lambda symbol: SimpleNamespace(history=lambda **kwargs: hist))
    item = YfinanceFetcher()._fetch_yf_ticker_data(yf, "fixture", "恒生科技指数", "HSTECH")
    assert item["current"] == 101.
    assert all(item[key] is None for key in ("prev_close", "change", "change_pct", "amplitude"))
    analyzer = _market()
    analyzer.data_manager = SimpleNamespace(get_main_indices=lambda region: [item])
    overview = analyzer.get_market_overview()
    table = analyzer._build_indices_block(overview)
    assert "N/A" in table and "+0.00%" not in table and "不代表收平" in table
    assert "N/A" in analyzer._build_review_prompt(overview, [])
    assert "N/A" in analyzer._generate_template_review(overview, [])
    snapshot = analyzer.build_market_light_snapshot(overview)
    assert snapshot["dimensions"]["index"]["available"] is False
    assert snapshot["data_quality"] == "unavailable"


def test_real_zero_change_remains_available():
    analyzer = _market()
    overview = MarketOverview(date="2026-10-23", indices=[MarketIndex(code="HSI", name="恒指", current=100.,
        change_pct=0., amplitude=0., prev_close=100.)])
    table = analyzer._build_indices_block(overview)
    assert "+0.00%" in table and "0.00%" in table
    assert analyzer.build_market_light_snapshot(overview)["dimensions"]["index"]["available"] is True


@pytest.mark.parametrize("dates,valid", [(["2026-10-21", "2026-10-23"], False),
                                       (["2026-10-23", "2026-10-26"], True)])
def test_daily_index_change_requires_adjacent_sessions_but_allows_weekends(dates, valid):
    hist = pd.DataFrame({"Open": [99., 100.], "High": [102., 103.], "Low": [98., 99.],
                         "Close": [100., 101.], "Volume": [1000., 1100.]}, index=pd.DatetimeIndex(dates))
    yf = SimpleNamespace(Ticker=lambda symbol: SimpleNamespace(history=lambda **kwargs: hist))
    item = YfinanceFetcher()._fetch_yf_ticker_data(yf, "fixture", "Fixture", "HSI", market="hk")
    if valid:
        assert item["change_pct"] == 1.
    else:
        assert item["change_pct"] is None and item["amplitude"] is None


def _bundle(*, statements=True):
    info = {"financialCurrency": "USD", "currency": "HKD", "totalRevenue": 900., "profitMargins": .1,
            "operatingCashflow": 80.}
    income = pd.DataFrame({pd.Timestamp("2026-06-30"): {"Total Revenue": 100., "Net Income": 20.}})
    cashflow = pd.DataFrame({pd.Timestamp("2025-12-31"): {"Operating Cash Flow": 25.},
                             pd.Timestamp("2026-03-31"): {"Operating Cash Flow": 30.}})
    ticker = SimpleNamespace(get_info=lambda: info, quarterly_income_stmt=income if statements else pd.DataFrame(),
                             quarterly_cashflow=cashflow if statements else pd.DataFrame(), dividends=pd.Series(dtype=float))
    with patch("yfinance.Ticker", return_value=ticker):
        return YfinanceFundamentalAdapter().get_fundamental_bundle("hk00100")


@pytest.mark.parametrize("language", ["zh", "en"])
@pytest.mark.parametrize("statements", [False, True])
def test_financial_currency_and_field_periods_reach_model_and_both_renderers(language, statements):
    bundle = _bundle(statements=statements)
    report = bundle["earnings"]["financial_report"]
    periods = report["field_periods"]
    if statements:
        assert periods["revenue"]["period_end"] == "2026-06-30"
        assert periods["operating_cash_flow"]["period_end"] == "2026-03-31"
        assert report["operating_cash_flow"] == 30.
    else:
        assert report["report_date"] is None
        assert periods["revenue"]["basis"] == "ttm"
        assert periods["net_profit_parent"]["basis"] == "estimated_ttm"
        assert report["net_profit_parent"] == 90.
    fundamental = {"earnings": {"data": bundle["earnings"]}, "growth": {"data": bundle["growth"]}}
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    context = {"code": "hk00100", "date": "2026-10-23", "today": {}, "fundamental_context": fundamental}
    prompt = analyzer._format_prompt(context, "Fixture", report_language=language)
    assert "USD" in prompt
    if not statements:
        assert "TTM" in prompt
    assert "ROE | None" not in prompt
    result = AnalysisResult(code="hk00100", name="Fixture", sentiment_score=50, trend_prediction="震荡",
        operation_advice="观望", report_language=language, fundamental_context=fundamental)
    notifier = NotificationService()
    for renderer in (False, True):
        with patch.object(notifier._config, "report_renderer_enabled", renderer):
            output = notifier.generate_dashboard_report([result])
        assert "USD" in output
        if not statements:
            assert "TTM" in output


@pytest.mark.parametrize("status", ["not_supported", "failed", "missing"])
@pytest.mark.parametrize("code", ["hk01810", "600519"])
def test_market_capability_does_not_silently_remove_fetch_failure_protection(code, status):
    result = AnalysisResult(code=code, name="Fixture", sentiment_score=75, trend_prediction="看多",
        operation_advice="买入", decision_type="buy", confidence_level="高")
    stabilize_decision_with_structure(result, None, {"capital_flow": {"status": status, "data": {}}})
    if code.startswith("hk") and status == "not_supported":
        assert result.decision_type == "buy" and result.sentiment_score == 75
        assert result.confidence_level == "中"
        assert any("资金流能力不支持" in text for text in result.dashboard["phase_decision"]["data_limitations"])
    else:
        assert result.decision_type == "hold"


@pytest.mark.parametrize("renderer", [False, True])
def test_daily_volume_is_not_mislabelled_as_realtime_volume_ratio(renderer):
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    snapshot = analyzer._build_market_snapshot({"date": "2026-10-23", "today": {"volume": 145.},
        "yesterday": {"volume": 100.}, "realtime": {"price": 10.}, "trend_analysis": {"volume_ratio_5d": 1.8}})
    result = AnalysisResult(code="hk01810", name="Fixture", sentiment_score=50, trend_prediction="震荡",
        operation_advice="观望", market_snapshot=snapshot, dashboard={"data_perspective": {
            "volume_analysis": {"volume_ratio": "invented", "volume_status": "放量", "turnover_rate": 18.}}})
    notifier = NotificationService()
    ground_volume_analysis(result)
    assert result.dashboard["data_perspective"]["volume_analysis"]["volume_ratio"] is None
    assert result.dashboard["data_perspective"]["volume_analysis"]["turnover_rate"] is None
    with patch.object(notifier._config, "report_renderer_enabled", renderer):
        output = notifier.generate_dashboard_report([result])
    assert "1.45×" in output and "1.80×" in output
    assert "独立指标" in output and "invented" not in output and "18.00%" not in output


@pytest.mark.parametrize("all_failed", [False, True])
def test_batch_failures_are_disclosed_without_entering_recommendations(all_failed):
    pipeline = batch_fixture.TestPipelineSingleStockNotify._build_batch_pipeline()
    pipeline.config.single_stock_notify = False
    failed = _make_result("000001", success=False)
    failed.error_message = "LLM returned empty response; fixture-private-provider-detail"
    pipeline.process_single_stock = MagicMock(side_effect=lambda code, **kwargs:
        failed if all_failed or code == "000001" else _make_result(code))
    results = pipeline.run(stock_codes=["000001", "600519"], send_notification=True)
    assert len(results) == (0 if all_failed else 1)
    pipeline._save_local_report.assert_called_once()
    pipeline._send_notifications.assert_called_once()
    output = pipeline._generate_aggregate_report(results, ReportType.SIMPLE)
    assert "本轮分析未全部完成" in output and "000001" in output
    assert "fixture-private-provider-detail" not in output
    assert "本轮分析未全部完成" not in pipeline._generate_aggregate_report(results, ReportType.SIMPLE, failure_codes=[])


@pytest.mark.parametrize("error,expected_calls", [("LLM returned empty response", 2), ("Authentication failed", 1)])
def test_empty_model_response_retries_are_bounded_and_auth_errors_are_not_retried(error, expected_calls):
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    analyzer._router = None
    analyzer._litellm_available = True
    analyzer._skill_instructions_override = ""
    analyzer._default_skill_policy_override = ""
    analyzer._use_legacy_default_prompt_override = True
    analyzer._config_override = Config(litellm_model="fixture/model", gemini_request_delay=0,
                                       report_integrity_enabled=False)
    with patch.object(analyzer, "_call_litellm", side_effect=_AllModelsFailedError(error)) as model:
        result = analyzer.analyze({"code": "hk01810", "stock_name": "Fixture", "today": {}})
    assert not result.success
    assert model.call_count == expected_calls


def test_empty_response_retry_can_recover_without_refetching_data():
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    analyzer._router = None
    analyzer._litellm_available = True
    analyzer._skill_instructions_override = ""
    analyzer._default_skill_policy_override = ""
    analyzer._use_legacy_default_prompt_override = True
    analyzer._config_override = Config(litellm_model="fixture/model", gemini_request_delay=0,
                                       report_integrity_enabled=False)
    response = json.dumps({"stock_name": "Fixture", "sentiment_score": 50, "operation_advice": "观望",
                           "trend_prediction": "震荡", "analysis_summary": "Fixture", "dashboard": {}})
    with patch.object(analyzer, "_call_litellm", side_effect=[_AllModelsFailedError("LLM returned empty response"),
                                                            (response, "fixture/model", {})]) as model:
        result = analyzer.analyze({"code": "hk01810", "stock_name": "Fixture", "today": {}})
    assert result.success and model.call_count == 2


@pytest.mark.parametrize("all_failed", [False, True])
def test_failure_only_email_group_receives_its_own_failure_and_other_groups_do_not(all_failed):
    pipeline = email_fixture.TestPipelineEmailGroupImageRouting()._build_pipeline()
    pipeline.config.stock_email_groups = [(["000001"], ["first@example.com"]),
                                         (["600519"], ["second@example.com"])]
    pipeline.analysis_failures = {"000001": "empty_response"}
    if all_failed:
        pipeline.analysis_failures["600519"] = "analysis_failed"
    pipeline.notifier._markdown_to_image_channels = set()
    with patch("src.md2img.markdown_to_image", return_value=None):
        pipeline._send_notifications([] if all_failed else [SimpleNamespace(code="600519")], ReportType.SIMPLE)
    messages = {tuple(call.kwargs["receivers"]): call.args[0]
                for call in pipeline.notifier.send_to_email.call_args_list}
    assert "000001" in messages[("first@example.com",)]
    assert "000001" not in messages[("second@example.com",)]
    assert ("本轮分析未全部完成" in messages[("second@example.com",)]) == all_failed


def test_all_failed_batch_still_sends_a_failure_report_without_email_groups():
    pipeline = email_fixture.TestPipelineEmailGroupImageRouting()._build_pipeline()
    pipeline.config.stock_email_groups = []
    pipeline.analysis_failures = {"000001": "empty_response"}
    pipeline.notifier._markdown_to_image_channels = set()
    with patch("src.md2img.markdown_to_image", return_value=None):
        pipeline._send_notifications([], ReportType.SIMPLE)
    pipeline.notifier.send_to_email.assert_called_once()
    assert "000001" in pipeline.notifier.send_to_email.call_args.args[0]


def test_missing_financial_currency_is_not_inferred_from_hk_trading_currency():
    ticker = SimpleNamespace(get_info=lambda: {"currency": "HKD", "totalRevenue": 100.},
        quarterly_income_stmt=pd.DataFrame(), quarterly_cashflow=pd.DataFrame(), dividends=pd.Series(dtype=float))
    with patch("yfinance.Ticker", return_value=ticker):
        bundle = YfinanceFundamentalAdapter().get_fundamental_bundle("hk00100")
    report = bundle["earnings"]["financial_report"]
    assert report["currency"] is None
    result = AnalysisResult(code="hk00100", name="Fixture", sentiment_score=50, trend_prediction="震荡",
        operation_advice="观望", fundamental_context={"earnings": {"data": bundle["earnings"]}})
    output = NotificationService().generate_dashboard_report([result])
    assert "财报币种：待确认" in output and "UNKNOWN" in output


def test_share_card_does_not_replace_unknown_quote_metrics_with_model_values():
    payload = {"code": "hk01810", "name": "Fixture", "sentiment_score": 50, "operation_advice": "观望",
        "market_snapshot": {"price": "10.00", "volume_ratio": "N/A", "turnover_rate": "N/A"},
        "dashboard": {"data_perspective": {"volume_analysis": {"volume_ratio": 9.99, "turnover_rate": 88.}}}}
    html = build_share_image_html("# Fixture\n\n## 数据透视", structured_payload=payload)
    assert "9.99" not in html and "88.00%" not in html
