"""omnichar_sdk.license against a fake Omnichar: every request goes through license._post."""

import http.client
import json
import logging
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from conftest import MANIFEST
from omnichar_sdk import (
    PRODUCTION_API,
    SANDBOX_API,
    CharError,
    LicenseError,
    LicenseRateLimited,
    LicenseRefused,
    LicenseUnreachable,
    read_license,
    verify,
)
from omnichar_sdk import license as lic
from omnichar_sdk.charfile import Manifest

SECRET = "lsk_live_" + "S3cr3tV4lu3" * 4 + "xyz"
TEST_SECRET = "lsk_test_" + "T" * 43
LICENSE_ID = "lic_live_Q2x8abcdefgh"
CHAR_ID = "7f1c2a9e-0000-4000-8000-000000000001"
TICKET = "lvt_" + "K" * 43
BLOCK = {
    "v": 1,
    "license_id": LICENSE_ID,
    "issuer": "omnichar",
    "key": LICENSE_ID,
    "env": "production",
    "kid": "live-2026-11-09",
    "signature": "kR0",
    "char_id": CHAR_ID,
    "scope": {"use": "advertising", "media": ["image"]},
}

REFUSALS = {
    "wrong_secret": "The license ID or secret is not right. Check the secret you were given.",
    "revoked": "This license was revoked by Omnicast Talents.",
    "expired": "This license expired on 12 March 2027.",
    "not_started": "This license starts on 1 December 2026.",
    "wrong_character": "This license is for a different character.",
    "output_not_allowed": "This license does not allow video.",
    "limit_reached": "This license has used all of its generations. Ask Omnicast Talents for more.",
}


def manifest(block=BLOCK):
    reserved = {} if block is None else {"license": block}
    return Manifest.from_json(dict(MANIFEST, reserved=reserved))


def valid(ttl=600, until="2099-01-01T00:00:00.000Z", **extra):
    body = {"status": "valid", "ticket": TICKET, "ttl_seconds": ttl, "valid_until": until}
    body.update({"owner_name": "Omnicast Talents"}, **extra)
    return (200, {}, json.dumps(body).encode())


def refused(status):
    return (403, {}, json.dumps({"status": status, "message": REFUSALS[status]}).encode())


class FakeOmnichar:
    """Answers each _post with the next scripted reply; an exception instance is raised."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, url, payload, timeout):
        self.calls.append((url, payload, timeout))
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    lic.clear_cache()
    monkeypatch.delenv(lic.ENV_OVERRIDE, raising=False)
    monkeypatch.setattr(lic, "_sleep", lambda seconds: None)
    yield
    lic.clear_cache()


def fake(monkeypatch, *answers):
    server = FakeOmnichar(*answers)
    monkeypatch.setattr(lic, "_post", server)
    return server


def check(output="image", secret=SECRET):
    found = read_license(manifest())
    return verify(found, CHAR_ID, secret, output, "unspecified", "comfyui", "0.2.1")


# --- reading the block ------------------------------------------------------------------------


def test_no_block_is_unlicensed():
    assert read_license(manifest(None)) is None


def test_a_block_reads_into_a_license_without_changing_it():
    found = read_license(manifest())
    assert (found.license_id, found.char_id, found.env, found.issuer) == (
        LICENSE_ID, CHAR_ID, "production", "omnichar",
    )
    assert dict(found.raw)["signature"] == "kR0"
    with pytest.raises(TypeError):
        found.raw["license_id"] = "lic_live_other000000"  # type: ignore[index]


@pytest.mark.parametrize(
    "block",
    [
        "lic_live_Q2x8abcdefgh",
        [],
        {},
        {"char_id": CHAR_ID},
        {"license_id": LICENSE_ID},
        {"license_id": "not-a-license", "char_id": CHAR_ID},
        {"license_id": LICENSE_ID, "char_id": ""},
        {"license_id": LICENSE_ID, "char_id": 7},
        {"license_id": LICENSE_ID + "\n", "char_id": CHAR_ID},
    ],
)
def test_a_damaged_block_is_refused_not_read_as_unlicensed(block):
    with pytest.raises(CharError, match="license block is unreadable"):
        read_license(manifest(block))


def test_license_errors_are_char_errors():
    assert issubclass(LicenseError, CharError)
    for cls in (LicenseRefused, LicenseRateLimited, LicenseUnreachable):
        assert issubclass(cls, LicenseError)


# --- where it asks ----------------------------------------------------------------------------


def test_the_secret_prefix_picks_the_environment():
    assert lic.api_base(SECRET) == PRODUCTION_API
    assert lic.api_base(TEST_SECRET) == SANDBOX_API
    assert PRODUCTION_API == "https://api.omnichar.org/v1"


@pytest.mark.parametrize(
    "override", ["http://127.0.0.1:8123/v1", "http://localhost:9/v1/", "http://[::1]:7/v1"]
)
def test_the_override_is_honoured_for_this_machine(monkeypatch, override):
    monkeypatch.setenv(lic.ENV_OVERRIDE, override)
    assert lic.api_base(SECRET) == override.rstrip("/")


@pytest.mark.parametrize(
    "override",
    [
        "https://evil.example/v1",
        "http://127.0.0.1@evil.example/v1",
        "http://localhost.evil.example/v1",
        "http://10.0.0.5/v1",
        "file:///etc/passwd",
        "ftp://localhost/v1",
        "http://[::1",
    ],
)
def test_the_override_is_ignored_for_any_other_host(monkeypatch, override):
    monkeypatch.setenv(lic.ENV_OVERRIDE, override)
    assert lic.api_base(SECRET) == PRODUCTION_API


# --- verify -----------------------------------------------------------------------------------


def test_a_valid_answer_gives_a_verdict(monkeypatch):
    server = fake(monkeypatch, valid())
    verdict = check()
    assert verdict.ticket == TICKET and verdict.ttl_seconds == 600
    assert verdict.owner_name == "Omnicast Talents"
    url, payload, timeout = server.calls[0]
    assert url == "https://api.omnichar.org/v1/licenses/verify"
    assert timeout == 10
    assert payload == {
        "license_id": LICENSE_ID,
        "secret": SECRET,
        "char_id": CHAR_ID,
        "output": "image",
        "model": "unspecified",
        "client": "comfyui",
        "client_version": "0.2.1",
    }


@pytest.mark.parametrize("status", sorted(REFUSALS))
def test_every_refusal_carries_the_servers_message(monkeypatch, status):
    fake(monkeypatch, refused(status))
    with pytest.raises(LicenseRefused) as excinfo:
        check()
    assert excinfo.value.status == status
    assert str(excinfo.value) == REFUSALS[status]


def test_too_many_attempts_is_not_retried(monkeypatch):
    body = json.dumps({"error": {"code": "rate_limited", "message": "Slow down."}}).encode()
    server = fake(monkeypatch, (429, {"Retry-After": "900"}, body))
    with pytest.raises(LicenseRateLimited) as excinfo:
        check()
    assert excinfo.value.retry_after == 900
    assert len(server.calls) == 1


@pytest.mark.parametrize(
    "answer",
    [
        urllib.error.URLError("no route"),
        TimeoutError("timed out"),
        ConnectionRefusedError(),
        http.client.IncompleteRead(b"partial"),
        (500, {}, b"oops"),
        (503, {}, b""),
        (422, {}, b'{"error":{"code":"unprocessable"}}'),
        (403, {}, b"<html>forbidden</html>"),
        (403, {}, b'{"error":{"code":"forbidden","message":"x"}}'),
        (200, {}, b"not json"),
        (200, {}, b"[]"),
        (200, {}, json.dumps({"status": "valid", "ttl_seconds": 600}).encode()),
        valid(ttl=True),
        valid(ttl=-5),
        valid(until="whenever"),
        valid(until="2099-01-01T00:00:00"),
        (200, {}, json.dumps({"status": "revoked"}).encode()),
        (302, {"Location": "https://elsewhere.example/"}, b""),
    ],
)
def test_no_usable_answer_is_unreachable(monkeypatch, answer):
    fake(monkeypatch, answer)
    with pytest.raises(LicenseUnreachable):
        check()


def test_a_verdict_is_reused_only_while_two_minutes_remain(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(lic, "_monotonic", lambda: clock[0])
    server = fake(monkeypatch, valid(ttl=600))
    check()
    clock[0] += 480  # exactly 120 s left
    check()
    assert len(server.calls) == 1
    clock[0] += 1  # 119 s left: a run starting now could outlive the ticket
    check()
    assert len(server.calls) == 2


def test_the_earlier_of_ttl_and_valid_until_ends_the_verdict(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(lic, "_monotonic", lambda: clock[0])
    monkeypatch.setattr(lic, "_wall", lambda: 4102444800.0 - 60)  # 60 s before 2100-01-01
    server = fake(monkeypatch, valid(ttl=600, until="2100-01-01T00:00:00.000Z"))
    check()
    clock[0] = 61
    check()
    assert len(server.calls) == 2


def test_the_cache_is_keyed_by_secret_and_output(monkeypatch):
    server = fake(monkeypatch, valid())
    check()
    check(output="video")
    check(secret="lsk_live_" + "Z" * 43)
    assert len(server.calls) == 3


def test_a_refusal_clears_that_licenses_cached_verdicts(monkeypatch):
    server = fake(monkeypatch, valid(), refused("output_not_allowed"), valid())
    check()
    with pytest.raises(LicenseRefused):
        check(output="video")
    check()
    assert len(server.calls) == 3


# --- usage ------------------------------------------------------------------------------------


def report(monkeypatch, *answers):
    server = fake(monkeypatch, *answers)
    lic.report_usage(TICKET, "image", "unspecified", "comfyui").join(5)
    return server


def test_usage_is_sent_once_when_accepted(monkeypatch, caplog):
    server = report(monkeypatch, (202, {}, b'{"accepted":true}'))
    url, payload, _ = server.calls[0]
    assert url == "https://api.omnichar.org/v1/licenses/usage"
    assert payload == {
        "ticket": TICKET, "output": "image", "model": "unspecified", "client": "comfyui"
    }
    assert len(server.calls) == 1 and not caplog.records


def test_usage_retries_a_network_error(monkeypatch, caplog):
    server = report(monkeypatch, urllib.error.URLError("down"), (202, {}, b"{}"))
    assert len(server.calls) == 2 and not caplog.records


def test_usage_gives_up_after_its_retries_with_one_line(monkeypatch, caplog):
    slept = []
    monkeypatch.setattr(lic, "_sleep", slept.append)
    server = report(monkeypatch, (503, {}, b""))
    assert len(server.calls) == 4
    assert slept == [1.0, 4.0, 16.0]
    assert len(caplog.records) == 1


@pytest.mark.parametrize("status", [401, 403, 429])
def test_a_refused_report_is_dropped_without_retrying(monkeypatch, caplog, status):
    server = report(monkeypatch, (status, {}, b'{"status":"invalid_ticket","message":"x"}'))
    assert len(server.calls) == 1
    assert len(caplog.records) == 1


def test_a_usage_429_waits_once_for_retry_after(monkeypatch, caplog):
    slept = []
    monkeypatch.setattr(lic, "_sleep", slept.append)
    server = report(monkeypatch, (429, {"Retry-After": "30"}, b"{}"), (202, {}, b"{}"))
    assert len(server.calls) == 2 and slept == [30.0] and not caplog.records


@pytest.mark.parametrize(
    "headers", [{"Retry-After": "30"}, {"Retry-After": "600"}, {}], ids=["twice", "long", "none"]
)
def test_a_usage_429_is_otherwise_dropped(monkeypatch, caplog, headers):
    slept = []
    monkeypatch.setattr(lic, "_sleep", slept.append)
    server = report(monkeypatch, (429, headers, b"{}"))
    assert len(server.calls) == (2 if headers.get("Retry-After") == "30" else 1)
    assert slept == ([30.0] if len(server.calls) == 2 else [])
    assert len(caplog.records) == 1


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (401, {"status": "invalid_ticket", "message": "Verify again."}),
        (403, {"status": "limit_reached", "message": "All generations used."}),
        (403, {"status": "output_not_allowed", "message": "No video."}),
        (429, {"status": "invalid_ticket", "message": "200 reports."}),
    ],
)
def test_a_refused_ticket_clears_the_verdict_and_says_why(monkeypatch, status, body):
    fake(monkeypatch, valid())
    check()
    heard = []
    server = fake(monkeypatch, (status, {"Retry-After": "1"}, json.dumps(body).encode()), valid())
    lic.report_usage(
        TICKET, "image", "unspecified", license_id=LICENSE_ID,
        on_refused=lambda s, m: heard.append((s, m)),
    ).join(5)
    assert heard == [(body["status"], body["message"])]
    assert len(server.calls) == 1
    check()
    assert len(server.calls) == 2  # verified again, not served from the cache


def test_a_label_with_a_trailing_newline_is_refused(monkeypatch):
    fake(monkeypatch, valid())
    with pytest.raises(ValueError):
        verify(read_license(manifest()), CHAR_ID, SECRET, "image", "flux\n")


def test_this_machine_is_reached_without_any_proxy(monkeypatch):
    """The loopback override uses its own opener, which has no proxy configured."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            data = valid()[2]
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    class Proxied:
        def open(self, *args, **kwargs):
            raise AssertionError("the proxied opener was used for this machine")

    monkeypatch.setattr(lic, "_opener", Proxied())
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv(lic.ENV_OVERRIDE, f"http://127.0.0.1:{server.server_port}/v1")
        assert check().ticket == TICKET
    finally:
        server.shutdown()
        server.server_close()
    # ProxyHandler({}) stands in for the default one that reads HTTP_PROXY, and adds no route.
    assert not any(getattr(h, "proxies", None) for h in lic._local_opener.handlers)
    assert not any(type(h).__name__ == "ProxyHandler" for h in lic._local_opener.handlers)


def test_usage_never_raises_into_the_caller(monkeypatch, caplog):
    report(monkeypatch, RuntimeError("anything at all"))
    assert len(caplog.records) == 1


# --- the secret stays put ---------------------------------------------------------------------


def test_no_secret_or_ticket_in_errors_logs_or_reprs(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv(lic.ENV_OVERRIDE, "https://evil.example/v1")
    seen = []
    answers = [
        refused("wrong_secret"),
        (429, {}, b""),
        urllib.error.URLError(f"echo {SECRET}"),
        (500, {}, SECRET.encode()),
        (200, {}, json.dumps({"status": "valid", "ticket": SECRET}).encode()),
    ]
    for answer in answers:
        fake(monkeypatch, answer)
        with pytest.raises(LicenseError) as excinfo:
            check()
        seen += [str(excinfo.value), repr(excinfo.value), repr(excinfo.value.__cause__)]
        assert excinfo.value.__cause__ is None
    fake(monkeypatch, valid())
    seen.append(repr(check()))
    report(monkeypatch, (500, {}, SECRET.encode()))
    text = "\n".join(seen) + caplog.text
    assert SECRET not in text and TICKET not in text


def test_a_redirect_is_not_followed(monkeypatch):
    """Against a real socket: a 307 that would resend the secret elsewhere stops the check."""
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            hits.append(self.path)
            self.send_response(307)
            self.send_header("Location", "/stolen")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv(lic.ENV_OVERRIDE, f"http://127.0.0.1:{server.server_port}/v1")
        with pytest.raises(LicenseUnreachable):
            check()
    finally:
        server.shutdown()
        server.server_close()
    assert hits == ["/v1/licenses/verify"]
