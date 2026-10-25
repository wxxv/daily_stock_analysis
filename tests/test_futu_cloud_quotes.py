"""Exercise Futu OAuth refresh and quote normalization through local HTTP."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs

import pytest

from data_provider.base import DataFetcherManager
from data_provider.futu_fetcher import FutuFetcher, _hk_provider_timestamp
from data_provider.futu_quote_client import FutuQuoteClient, FutuQuoteError
from src.config import Config
from src.core.config_registry import get_field_definition


@pytest.fixture
def quote_server(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    state = {
        "calls": [],
        "scope": "quote:read accid:*",
        "snapshot_statuses": [],
        "token_status": 200,
        "expires_in": 7200,
        "snapshot": {
            "code": "HK.00700", "sc_name": "腾讯控股", "last_price": 386.0,
            "prev_close_price": 382.4, "open_price": 382.4, "high_price": 390.0,
            "low_price": 380.0, "volume": 33161500, "turnover": 12730000000.0,
            "turnover_rate": 0.365, "volume_ratio": 1.03, "pe_ttm_ratio": 13.6,
            "update_time": 1792742878000, "data_date": "2026-10-23",
        },
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode()
            body = parse_qs(raw) if self.path == "/oauth2/token" else json.loads(raw)
            state["calls"].append((self.path, body, self.headers.get("Authorization")))
            status = 200
            if self.path == "/oauth2/token":
                status = state["token_status"]
                payload = {"access_token": f"access-{len(state['calls'])}",
                           "expires_in": state["expires_in"], "scope": state["scope"]}
            elif self.path == "/api/v1.0/quote/snapshot":
                status = state["snapshot_statuses"].pop(0) if state["snapshot_statuses"] else 200
                payload = {"ret_code": 0, "data": {"snapshot_list": [state["snapshot"]]}}
            else:
                status, payload = 404, {}
            if status != 200:
                payload = {"error": "refresh-test-secret"}
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.05), daemon=True)
    worker.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def _client(server):
    return FutuQuoteClient("test-client", "refresh-test-secret", api_base_url=server["url"])


def _fetcher(server):
    config = SimpleNamespace(futu_opend_host=None, futu_oauth_client_id="test-client",
                             futu_oauth_refresh_token="refresh-test-secret")
    with patch.dict(os.environ, {}, clear=True), patch("src.config.get_config", return_value=config):
        fetcher = FutuFetcher()
    fetcher._http_client._base_url = server["url"]
    return fetcher


def test_cloud_quote_refreshes_once_and_reuses_token(quote_server):
    client = _client(quote_server)
    try:
        for _ in range(2):
            assert client.get_snapshot("HK.00700")["volume_ratio"] == 1.03
        calls = quote_server["calls"]
        assert [call[0] for call in calls].count("/oauth2/token") == 1
        assert calls[0][1] == {"client_id": ["test-client"], "grant_type": ["refresh_token"],
                               "refresh_token": ["refresh-test-secret"]}
        assert all(call[1] == {"code_list": ["HK.00700"]} and call[2].startswith("Bearer ")
                   for call in calls if call[0].endswith("snapshot"))
    finally:
        client.close()


def test_expired_token_is_refreshed(quote_server, monkeypatch):
    now = [100.0]
    monkeypatch.setattr("data_provider.futu_quote_client.time", SimpleNamespace(monotonic=lambda: now[0]))
    quote_server["expires_in"] = 1
    client = _client(quote_server)
    client.get_snapshot("HK.00700")
    now[0] = 102.0
    client.get_snapshot("HK.00700")
    assert sum(path == "/oauth2/token" for path, *_ in quote_server["calls"]) == 2
    client.close()


@pytest.mark.parametrize("statuses,expected_refreshes,raises", [([401, 200], 2, False), ([401, 401], 2, True),
                                                              ([403], 1, True), ([429], 1, True), ([307], 1, True)])
def test_only_401_retries_once_without_exposing_response_bodies(quote_server, statuses, expected_refreshes, raises):
    quote_server["snapshot_statuses"] = statuses
    client = _client(quote_server)
    if raises:
        with pytest.raises(FutuQuoteError) as error:
            client.get_snapshot("HK.00700")
        assert "refresh-test-secret" not in str(error.value)
    else:
        assert client.get_snapshot("HK.00700")["code"] == "HK.00700"
    assert sum(path == "/oauth2/token" for path, *_ in quote_server["calls"]) == expected_refreshes
    client.close()


@pytest.mark.parametrize("scope", ["trade:read quote:read", "quote:read trade:write", "quote:write", ""])
def test_non_read_only_grants_are_rejected_before_snapshot(quote_server, scope):
    quote_server["scope"] = scope
    client = _client(quote_server)
    with pytest.raises(FutuQuoteError, match="quote-read-only"):
        client.get_snapshot("HK.00700")
    assert len(quote_server["calls"]) == 1
    client.close()


def test_revoked_refresh_token_is_not_hidden_as_an_empty_quote(quote_server):
    quote_server["token_status"] = 400
    client = _client(quote_server)
    with pytest.raises(FutuQuoteError, match="HTTP 400") as error:
        client.get_snapshot("HK.00700")
    assert "refresh-test-secret" not in str(error.value)
    assert len(quote_server["calls"]) == 1
    client.close()


def test_different_stock_is_never_used_as_requested_snapshot(quote_server):
    quote_server["snapshot"]["code"] = "HK.09988"
    client = _client(quote_server)
    with pytest.raises(FutuQuoteError, match="omitted requested symbol"):
        client.get_snapshot("HK.00700")
    client.close()


def test_fetcher_maps_real_units_timestamp_and_unknown_fields(quote_server):
    fetcher = _fetcher(quote_server)
    quote = fetcher.get_realtime_quote("HK00700")
    assert quote.name == "腾讯控股"
    assert quote.source.value == "futu"
    assert quote.turnover_rate == 0.365
    assert quote.volume_ratio == 1.03
    assert quote.change_pct == pytest.approx((386 / 382.4 - 1) * 100)
    assert quote.provider_timestamp.endswith("+08:00")
    quote_server["snapshot"].pop("volume_ratio")
    assert fetcher.get_realtime_quote("HK00700").volume_ratio is None
    assert fetcher.is_available_for_request("realtime")
    assert not fetcher.is_available_for_request("daily")
    assert not fetcher.is_available_for_request("fundamental")
    assert DataFetcherManager._filter_fetchers_by_capability([fetcher], "daily") == []
    fetcher.close()


def test_unconfigured_cloud_does_not_enable_a_provider():
    with patch.dict(os.environ, {}, clear=True), patch("src.config.get_config", return_value=SimpleNamespace()):
        assert not FutuFetcher.has_configured_http_credentials()
        assert not FutuFetcher()._is_available()


def test_cloud_timestamp_does_not_treat_milliseconds_as_seconds():
    assert _hk_provider_timestamp(1792742878000) == "2026-10-23T16:07:58+08:00"
    for value in (0, -1, float("nan"), float("inf")):
        assert _hk_provider_timestamp(value) is None


def test_cloud_config_loads_and_masks_refresh_token():
    with patch("src.config.setup_env"), patch.dict(os.environ, {
        "FUTU_OAUTH_CLIENT_ID": " test-client ", "FUTU_OAUTH_REFRESH_TOKEN": " refresh-test-secret ",
    }, clear=True):
        config = Config._load_from_env()
    assert config.futu_oauth_client_id == "test-client"
    assert config.futu_oauth_refresh_token == "refresh-test-secret"
    assert "refresh-test-secret" not in repr(config)
    field = get_field_definition("FUTU_OAUTH_REFRESH_TOKEN")
    assert field["is_sensitive"] and field["ui_control"] == "password"


def test_partial_credentials_are_reported_without_disclosing_token():
    config = Config(futu_oauth_refresh_token="refresh-test-secret")
    issues = [issue for issue in config.validate_structured() if issue.field == "FUTU_OAUTH_CLIENT_ID"]
    assert issues and issues[0].severity == "warning"
    assert "refresh-test-secret" not in issues[0].message


def test_actions_maps_refresh_token_only_from_secrets():
    from pathlib import Path
    import yaml

    workflow = yaml.load(
        (Path(__file__).resolve().parents[1] / ".github/workflows/00-daily-analysis.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    env = next(step["env"] for step in workflow["jobs"]["analyze"]["steps"] if step.get("name") == "执行股票分析")
    assert env["FUTU_OAUTH_REFRESH_TOKEN"] == "${{ secrets.FUTU_OAUTH_REFRESH_TOKEN }}"
    assert "vars.FUTU_OAUTH_CLIENT_ID" in env["FUTU_OAUTH_CLIENT_ID"]
    assert "vars.FUTU_HK_REALTIME_SOURCE_PRIORITY" in env["FUTU_HK_REALTIME_SOURCE_PRIORITY"]


@pytest.mark.parametrize("renderer", [False, True])
def test_cloud_metrics_reach_report_without_model_substitution(quote_server, renderer):
    from src.analyzer import AnalysisResult, GeminiAnalyzer
    from src.notification import NotificationService
    from src.reports.volume_context import ground_volume_analysis

    fetcher = _fetcher(quote_server)
    quote = fetcher.get_realtime_quote("HK00700")
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    snapshot = analyzer._build_market_snapshot({"date": "2026-10-23", "realtime": quote.to_dict()})
    assert snapshot["volume_ratio"] != "N/A"
    assert snapshot["turnover_rate"] != "N/A"
    result = AnalysisResult(
        code="HK00700", name="腾讯控股", sentiment_score=50, trend_prediction="震荡", operation_advice="观望",
        market_snapshot=snapshot, dashboard={"data_perspective": {"volume_analysis": {
            "volume_ratio": 88.8, "turnover_rate": 99.9,
        }}},
    )
    ground_volume_analysis(result)
    notifier = NotificationService()
    with patch.object(notifier._config, "report_renderer_enabled", renderer):
        report = notifier.generate_dashboard_report([result])
    assert "1.03" in report
    assert "0.36%" in report or "0.37%" in report
    assert "88.8" not in report and "99.9" not in report
    fetcher.close()
