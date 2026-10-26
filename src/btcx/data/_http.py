"""Cliente HTTP compartido: sesión requests + reintentos tenacity."""

from __future__ import annotations

from pathlib import Path

import requests
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

_USER_AGENT = "btcx/0.2 (data pipeline; +https://github.com/Kartoffel156/btc-exposure-net)"
_TIMEOUT = 30


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, requests.HTTPError):
        code = exc.response.status_code if exc.response is not None else 0
        return code in (429, 500, 502, 503, 504)
    return isinstance(exc, requests.ConnectionError)


class HttpClient:
    def __init__(self, raw_dir: Path | None = None) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})
        self._raw_dir = raw_dir

    @retry(
        retry=retry_if_exception(_retryable),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def get(self, url: str, **kwargs: object) -> requests.Response:
        kwargs.setdefault("timeout", _TIMEOUT)
        resp = self._session.get(url, **kwargs)  # type: ignore[arg-type]
        resp.raise_for_status()
        return resp

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def make_http_client(raw_dir: Path | None = None) -> HttpClient:
    return HttpClient(raw_dir=raw_dir)
