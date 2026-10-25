"""Futu cloud quote snapshots authenticated by a read-only OAuth grant."""

import threading
import time
from typing import Any, Dict, Optional

import requests


class FutuQuoteError(RuntimeError):
    """A sanitized provider error, safe to include in fallback diagnostics."""


class FutuQuoteClient:
    """Refresh a quote grant in memory and call only the snapshot endpoint."""

    def __init__(
        self,
        client_id: str,
        refresh_token: str,
        *,
        timeout: float = 10.0,
        api_base_url: str = "https://webapi.futunn.com",
    ) -> None:
        self._client_id = client_id
        self._refresh_token = refresh_token
        self._timeout = timeout
        self._base_url = api_base_url.rstrip("/")
        self._session = requests.Session()
        self._token_lock = threading.Lock()
        self._access_token: Optional[str] = None
        self._expires_at = 0.0

    def _post(self, path: str, **kwargs) -> Dict[str, Any]:
        try:
            response = self._session.post(
                self._base_url + path,
                timeout=self._timeout,
                allow_redirects=False,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise FutuQuoteError(f"Futu {path} network error: {type(exc).__name__}") from None
        if not 200 <= response.status_code < 300:
            error = FutuQuoteError(f"Futu {path} returned HTTP {response.status_code}")
            error.status_code = response.status_code
            raise error
        try:
            payload = response.json()
        except ValueError:
            raise FutuQuoteError(f"Futu {path} returned invalid JSON") from None
        if not isinstance(payload, dict):
            raise FutuQuoteError(f"Futu {path} returned an invalid response")
        return payload

    def _get_access_token(self) -> str:
        with self._token_lock:
            if self._access_token and time.monotonic() < self._expires_at:
                return self._access_token
            payload = self._post(
                "/oauth2/token",
                data={
                    "grant_type": "refresh_token",
                    "client_id": self._client_id,
                    "refresh_token": self._refresh_token,
                },
            )
            scopes = set(str(payload.get("scope") or "").split())
            if "quote:read" not in scopes or any(
                scope != "quote:read" and not scope.startswith("accid:") for scope in scopes
            ):
                raise FutuQuoteError("Futu OAuth requires a quote-read-only grant; reauthorize quote:read")
            token = payload.get("access_token")
            try:
                expires_in = int(payload.get("expires_in", 0))
            except (TypeError, ValueError, OverflowError):
                expires_in = 0
            if not isinstance(token, str) or not token or expires_in <= 0:
                raise FutuQuoteError("Futu OAuth returned no usable access token or expiry")
            self._access_token = token
            self._expires_at = time.monotonic() + max(0, expires_in - min(60, expires_in / 10))
            return token

    def get_snapshot(self, symbol: str) -> Dict[str, Any]:
        for attempt in range(2):
            token = self._get_access_token()
            try:
                payload = self._post(
                    "/api/v1.0/quote/snapshot",
                    json={"code_list": [symbol]},
                    headers={"Authorization": f"Bearer {token}"},
                )
            except FutuQuoteError as exc:
                if attempt == 0 and getattr(exc, "status_code", None) == 401:
                    with self._token_lock:
                        if self._access_token == token:
                            self._access_token = None
                            self._expires_at = 0.0
                    continue
                raise
            if payload.get("ret_code") != 0:
                raise FutuQuoteError("Futu snapshot request was rejected; check quote permissions")
            data = payload.get("data")
            if not isinstance(data, dict):
                raise FutuQuoteError("Futu snapshot returned an invalid data block")
            rows = data.get("snapshot_list", [])
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict) and row.get("code") == symbol:
                        return row
            raise FutuQuoteError(f"Futu snapshot omitted requested symbol {symbol}")
        raise FutuQuoteError("Futu snapshot authentication failed after one refresh")

    def close(self) -> None:
        self._session.close()
