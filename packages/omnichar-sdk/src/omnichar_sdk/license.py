# SPDX-License-Identifier: Apache-2.0
"""Check a licensed character with Omnichar before use; only the server decides if it is valid.

Nothing here signs, edits or verifies a license offline. ``reserved.license`` is read, never
written, and every URL comes from this module or a localhost override, never from the file.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import os
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from .charfile import CharError, Manifest

logger = logging.getLogger("omnichar_sdk")

PRODUCTION_API = "https://api.omnichar.org/v1"
#: The same host: an ``lsk_test_`` secret is what selects the sandbox, so the rule stays named.
SANDBOX_API = PRODUCTION_API

#: Honoured only for a loopback host, so a stray environment variable cannot send a secret away.
ENV_OVERRIDE = "OMNICHAR_LICENSE_API"
_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})

TIMEOUT_SECONDS = 10.0
OUTPUTS = ("image", "video")

#: The shape Cloud issues; anything else in a file is damage, not a license to try.
_LICENSE_ID = re.compile(r"^lic_(live|test)_[A-Za-z0-9_-]{10,60}$")
#: What verify accepts for model, client and client_version.
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._:/@+-]*$")
_MAX_BODY = 64 * 1024
_MAX_MESSAGE = 500
#: Usage retries, in seconds; a report is dropped after the last one.
USAGE_BACKOFF = (1.0, 4.0, 16.0)
#: A 429 on a usage report is retried once, after the server's wait up to this many seconds.
USAGE_MAX_WAIT = 60
#: A cached verdict is reused only with this much left, so a run never starts on a dying ticket.
REUSE_MARGIN = 120.0

# Seams for tests: every request goes through _post, and time through these two.
_monotonic = time.monotonic
_wall = time.time
_sleep = time.sleep


class LicenseError(CharError):
    """A licensed character could not be cleared for use. A `CharError`, so old handlers apply."""


class LicenseRefused(LicenseError):
    """Omnichar refused the license; ``message`` is written for the user."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class LicenseRateLimited(LicenseError):
    """Too many checks; ``retry_after`` is the server's wait in seconds, when it gave one."""

    def __init__(self, retry_after: int | None) -> None:
        super().__init__("Too many license checks. Try again in a few minutes.")
        self.retry_after = retry_after


class LicenseUnreachable(LicenseError):
    """No usable answer from Omnichar: offline, timed out, a server error or a garbled reply."""


@dataclass(frozen=True)
class License:
    """The ``reserved.license`` block of a ``.char``, as read. ``raw`` is a read-only copy."""

    license_id: str
    char_id: str
    env: str
    issuer: str
    raw: Mapping[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True)
class Verdict:
    """A valid answer. ``expires_at`` is a ``time.monotonic`` deadline for reusing it."""

    ticket: str = field(repr=False)
    ttl_seconds: int
    valid_until: str
    owner_name: str
    expires_at: float


def read_license(manifest: Manifest) -> License | None:
    """The license a manifest carries, or None. A damaged block raises rather than reading as
    unlicensed, so a broken file cannot slip past the check."""
    block = manifest.reserved.get("license")
    if block is None:
        return None
    unreadable = CharError("This character's license block is unreadable.")
    if not isinstance(block, dict):
        raise unreadable
    license_id = block.get("license_id")
    char_id = block.get("char_id")
    if not isinstance(license_id, str) or not _LICENSE_ID.fullmatch(license_id):
        raise unreadable
    if not isinstance(char_id, str) or not char_id or len(char_id) > 200:
        raise unreadable
    env = block.get("env")
    issuer = block.get("issuer")
    return License(
        license_id=license_id,
        char_id=char_id,
        env=env if isinstance(env, str) else "",
        issuer=issuer if isinstance(issuer, str) else "",
        raw=MappingProxyType(json.loads(json.dumps(block))),
    )


def _is_loopback(url: str) -> bool:
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and host in _LOOPBACK


_override_warned = False


def api_base(secret: str = "") -> str:
    """Where to verify: production, or the sandbox for an ``lsk_test_`` secret."""
    global _override_warned
    override = os.environ.get(ENV_OVERRIDE, "").strip()
    if override:
        if _is_loopback(override):
            return override.rstrip("/")
        if not _override_warned:
            _override_warned = True
            logger.warning("%s is ignored: it may only point at this machine.", ENV_OVERRIDE)
    return SANDBOX_API if secret.startswith("lsk_test_") else PRODUCTION_API


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    # A redirect would carry the secret somewhere this module did not choose.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_opener = urllib.request.build_opener(
    urllib.request.HTTPSHandler(context=ssl.create_default_context()), _NoRedirect()
)
#: For the loopback override: no proxy ever sees a plain-HTTP request to this machine.
_local_opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({}),
    urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    _NoRedirect(),
)


def _post(
    url: str, payload: dict[str, Any], timeout: float
) -> tuple[int, Mapping[str, str], bytes]:
    """POST JSON; returns (status, headers, body) for any HTTP answer, raises on no answer."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        opener = _local_opener if _is_loopback(url) else _opener
        with opener.open(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read(_MAX_BODY)
    except urllib.error.HTTPError as error:
        with error:
            body = error.read(_MAX_BODY) if error.fp is not None else b""
            return error.code, dict(error.headers or {}), body


def _json(body: bytes) -> dict[str, Any] | None:
    try:
        value = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _clean(text: str) -> str:
    """A server message made safe to show: printable, one paragraph, bounded."""
    return "".join(c if c.isprintable() else " " for c in text).strip()[:_MAX_MESSAGE]


def _until(valid_until: str) -> float | None:
    """Seconds from now until an ISO timestamp, or None if it does not parse."""
    try:
        moment = datetime.fromisoformat(valid_until.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        return None
    return moment.timestamp() - _wall()


def _retry_after(headers: Mapping[str, str]) -> int | None:
    for key, value in headers.items():
        if key.lower() == "retry-after":
            try:
                return max(0, int(value.strip()))
            except ValueError:
                return None
    return None


_cache: dict[tuple[str, str, str, str], Verdict] = {}
_cache_lock = threading.Lock()


def clear_cache(license_id: str | None = None) -> None:
    """Forget cached verdicts: one license's, or every one."""
    with _cache_lock:
        if license_id is None:
            _cache.clear()
            return
        for key in [k for k in _cache if k[0] == license_id]:
            del _cache[key]


def _check_label(name: str, value: str, limit: int) -> None:
    if not isinstance(value, str) or len(value) > limit or not _LABEL.fullmatch(value):
        raise ValueError(f"{name} must be 1 to {limit} letters, digits, spaces or . _ : / @ + -")


def verify(
    license: License,
    char_id: str,
    secret: str,
    output: str,
    model: str,
    client: str = "comfyui",
    client_version: str = "0",
) -> Verdict:
    """Ask Omnichar whether ``secret`` unlocks ``license`` for ``output``; cached until it ends."""
    if output not in OUTPUTS:
        raise ValueError(f"output must be one of {', '.join(OUTPUTS)}")
    _check_label("model", model, 80)
    _check_label("client", client, 40)
    _check_label("client_version", client_version, 40)
    if not isinstance(secret, str) or not secret or len(secret) > 100:
        raise ValueError("secret must be 1 to 100 characters")

    key = (license.license_id, char_id, output, hashlib.sha256(secret.encode()).hexdigest())
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None and cached.expires_at - _monotonic() >= REUSE_MARGIN:
            return cached
        _cache.pop(key, None)

    payload = {
        "license_id": license.license_id,
        "secret": secret,
        "char_id": char_id,
        "output": output,
        "model": model,
        "client": client,
        "client_version": client_version,
    }
    try:
        status, headers, body = _post(
            f"{api_base(secret)}/licenses/verify", payload, TIMEOUT_SECONDS
        )
    except (OSError, ValueError, http.client.HTTPException):
        # URLError, timeouts and refused connections are all OSError; the cause is not chained,
        # so nothing about the request rides along in a traceback.
        raise LicenseUnreachable("Can't reach Omnichar to check this license.") from None

    if status == 200:
        verdict = _verdict(_json(body))
        if verdict is None:
            raise LicenseUnreachable("Omnichar sent an answer this version cannot read.")
        with _cache_lock:
            _cache[key] = verdict
        return verdict
    if status == 403:
        data = _json(body) or {}
        refusal, message = data.get("status"), data.get("message")
        if isinstance(refusal, str) and isinstance(message, str) and _clean(message):
            clear_cache(license.license_id)
            raise LicenseRefused(refusal, _clean(message))
    if status == 429:
        raise LicenseRateLimited(_retry_after(headers))
    raise LicenseUnreachable(f"Omnichar could not check this license (HTTP {status}).")


def _verdict(data: dict[str, Any] | None) -> Verdict | None:
    if data is None or data.get("status") != "valid":
        return None
    ticket = data.get("ticket")
    ttl = data.get("ttl_seconds")
    valid_until = data.get("valid_until")
    owner = data.get("owner_name", "")
    if not isinstance(ticket, str) or not ticket or len(ticket) > 512:
        return None
    if isinstance(ttl, bool) or not isinstance(ttl, int) or ttl <= 0:
        return None
    if not isinstance(valid_until, str):
        return None
    remaining = _until(valid_until)
    if remaining is None:
        return None
    return Verdict(
        ticket=ticket,
        ttl_seconds=ttl,
        valid_until=valid_until,
        owner_name=_clean(owner) if isinstance(owner, str) else "",
        expires_at=_monotonic() + max(0.0, min(float(ttl), remaining)),
    )


def report_usage(
    ticket: str,
    output: str,
    model: str,
    client: str = "comfyui",
    *,
    base: str | None = None,
    license_id: str | None = None,
    on_refused: Callable[[str, str], None] | None = None,
) -> threading.Thread:
    """Report one generation in the background. Never raises and never blocks the caller; a
    report that cannot be delivered is dropped with one log line.

    When Omnichar refuses the ticket (401, 403, or 429 ``invalid_ticket``), the cached verdicts
    for ``license_id`` are dropped, so the next check asks again, and ``on_refused(status,
    message)`` is told why."""
    url = f"{base or api_base()}/licenses/usage"
    payload = {"ticket": ticket, "output": output, "model": model, "client": client}
    thread = threading.Thread(
        target=_deliver,
        args=(url, payload, license_id, on_refused),
        name="omnichar-usage",
        daemon=True,
    )
    thread.start()
    return thread


def _refused(
    status: int,
    body: bytes,
    license_id: str | None,
    on_refused: Callable[[str, str], None] | None,
) -> bool:
    """Whether a usage answer says the ticket no longer works; if so, forget the verdict."""
    data = _json(body) or {}
    refusal, message = data.get("status"), data.get("message")
    if status == 429 and refusal != "invalid_ticket":
        return False
    if status not in (401, 403, 429):
        return False
    if license_id is not None:
        clear_cache(license_id)
    if on_refused is not None:
        text = _clean(message) if isinstance(message, str) else ""
        on_refused(refusal if isinstance(refusal, str) else f"http_{status}", text)
    return True


def _deliver(
    url: str,
    payload: dict[str, Any],
    license_id: str | None = None,
    on_refused: Callable[[str, str], None] | None = None,
) -> None:
    try:
        reason = "no answer"
        backoff = list(USAGE_BACKOFF)
        pause = 0.0
        waited = False
        while True:
            if pause:
                _sleep(pause)
            try:
                status, headers, body = _post(url, payload, TIMEOUT_SECONDS)
            except (OSError, ValueError, http.client.HTTPException):
                reason = "no answer"
                if not backoff:
                    break
                pause = backoff.pop(0)
                continue
            if 200 <= status < 300:
                return
            reason = f"HTTP {status}"
            if _refused(status, body, license_id, on_refused):
                break
            if status == 429:
                wait = _retry_after(headers)
                if waited or wait is None or wait > USAGE_MAX_WAIT:
                    break
                waited = True
                pause = float(wait)
                continue
            if status < 500 or not backoff:
                break
            pause = backoff.pop(0)
        logger.warning("Omnichar: a licensed generation was not reported (%s).", reason)
    except Exception:  # a report must never surface in the caller's process
        logger.warning("Omnichar: a licensed generation was not reported.")
