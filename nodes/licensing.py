# SPDX-License-Identifier: GPL-3.0-or-later
"""License secrets on this machine, the check Load Character runs, and the routes the UI uses."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import folder_paths
from omnichar_sdk import Character, CharError

from . import folders
from .common import licensed, mark_verified

logger = logging.getLogger("omnichar")

try:
    from omnichar_sdk import license as sdk_license
except ImportError:  # an SDK before 0.1.2: unlicensed characters still work
    sdk_license = None
    logger.warning(
        "Omnichar licensed characters are not available: this omnichar-sdk is too old.\n"
        "Update it with: pip install -U 'omnichar-sdk>=0.1.2,<0.2'\n"
        "If ComfyUI runs in its own environment, use that environment's pip."
    )

#: Sent as client_version; a test keeps it equal to pyproject.toml.
PACK_VERSION = "0.2.1"
CLIENT = "comfyui"
#: No Omnichar node knows the checkpoint, and verify requires a model.
MODEL = "unspecified"

OLD_SDK = "Update omnichar-sdk to use licensed characters."
UNREACHABLE = "Can't reach OmniChar to check this license. Connect and try again."
RATE_LIMITED = "Too many attempts. Try again in a few minutes."
REPORT_REFUSED = "Omnichar refused the last usage report. Queue again to check the license."

#: From the SDK when it has them, so the widget and verify never disagree.
OUTPUTS: tuple[str, ...] = tuple(sdk_license.OUTPUTS) if sdk_license else ("image", "video")
#: How long a cleared character stays usable downstream. Load re-runs on every queue (IS_CHANGED
#: is NaN), so this only has to outlast one run, including a slow model load.
RUN_GRACE_SECONDS = 3600.0

LICENSE_ID = re.compile(r"^lic_(live|test)_[A-Za-z0-9_-]{10,60}$")
SECRET = re.compile(r"^lsk_(live|test)_[A-Za-z0-9_-]{20,80}$")
MAX_BODY = 4096
_MAX_NAME = 255


def missing_secret(name: str) -> str:
    return f"{name}.char is licensed. Enter the license secret from the share email."


def label(char: str, char_path: str, path: Path) -> str:
    """The name the user picked, not the symlink target it resolves to."""
    return path.stem if char_path.strip() else Path(char).stem


# --- the secret store -------------------------------------------------------------------------

_lock = threading.Lock()


def _store_path() -> Path:
    return Path(folder_paths.get_user_directory()) / "omnichar" / "licenses.json"


def _read_store() -> dict[str, str]:
    try:
        data = json.loads(_store_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        k: v
        for k, v in data.items()
        if isinstance(k, str)
        and isinstance(v, str)
        and LICENSE_ID.fullmatch(k)
        and SECRET.fullmatch(v)
    }


def _write_store(data: dict[str, str]) -> None:
    """Atomic: a temp file readable only by this user, fsynced, then renamed over the old one."""
    target = _store_path()
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".licenses.", suffix=".tmp")
    try:
        os.chmod(tmp, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def secret_for(license_id: str) -> str | None:
    with _lock:
        return _read_store().get(license_id)


def store_secret(license_id: str, secret: str) -> None:
    with _lock:
        data = _read_store()
        data[license_id] = secret
        _write_store(data)
    _forget_checks(license_id)


def forget_secret(license_id: str) -> None:
    with _lock:
        data = _read_store()
        if data.pop(license_id, None) is not None:
            _write_store(data)
    _forget_checks(license_id)


# --- what the UI shows ------------------------------------------------------------------------

_status: dict[str, dict[str, Any]] = {}


def _forget_checks(license_id: str) -> None:
    with _lock:
        _status.pop(license_id, None)
    if sdk_license is not None:
        sdk_license.clear_cache(license_id)


def _record(license_id: str, **fields: Any) -> None:
    with _lock:
        _status[license_id] = fields


def status_of(license_id: str) -> dict[str, Any]:
    out: dict[str, Any] = {"stored": secret_for(license_id) is not None}
    with _lock:
        out.update(_status.get(license_id, {}))
    return out


# --- opening and clearing a character ---------------------------------------------------------

#: (path, mtime_ns, size) -> Character; opened once, not per queue.
_opened: OrderedDict[tuple[str, int, int], Character] = OrderedDict()
_MAX_OPENED = 32


def open_cached(path: Path) -> Character:
    stat = os.stat(path)
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    with _lock:
        hit = _opened.get(key)
        if hit is not None:
            _opened.move_to_end(key)
            return hit
    opened = Character.open(path)
    with _lock:
        _opened[key] = opened
        while len(_opened) > _MAX_OPENED:
            _opened.popitem(last=False)
    return opened


def is_licensed(path: Path) -> bool:
    """Whether a file carries a license; an unreadable file is left to the loader to report."""
    try:
        return licensed(open_cached(path)) is not None
    except (CharError, OSError):
        return False


def precheck(char: Character, name: str) -> str | None:
    """The reason a licensed character cannot be queued yet, without going online."""
    if licensed(char) is None:
        return None
    if sdk_license is None:
        return OLD_SDK
    try:
        found = sdk_license.read_license(char.manifest)
    except CharError as error:
        return str(error)
    if found is None or secret_for(found.license_id) is None:
        return missing_secret(name)
    return None


def _current_run() -> str | None:
    """The queued prompt being executed, where this ComfyUI exposes it."""
    try:
        from comfy_execution.utils import get_executing_context

        context = get_executing_context()
    except Exception:  # older ComfyUI, or not inside a run
        return None
    prompt_id = getattr(context, "prompt_id", None) if context is not None else None
    return str(prompt_id) if prompt_id else None


_reported: dict[str, str] = {}


def _first_report(license_id: str) -> bool:
    """One report per license per queued run, even with two Load nodes on the same character."""
    run = _current_run()
    if run is None:
        return True
    with _lock:
        if _reported.get(license_id) == run:
            return False
        _reported[license_id] = run
    return True


def unlock(char: Character, name: str, output: str) -> Character:
    """Verify a licensed character with Omnichar, mark it usable for this run, report the use."""
    if sdk_license is None:
        raise CharError(OLD_SDK)
    found = sdk_license.read_license(char.manifest)
    if found is None:
        return char
    license_id = found.license_id
    secret = secret_for(license_id)
    if secret is None:
        raise CharError(missing_secret(name))
    try:
        verdict = sdk_license.verify(
            found, found.char_id, secret, output, MODEL, CLIENT, PACK_VERSION
        )
    except sdk_license.LicenseRefused as error:
        _record(license_id, last_error={"status": error.status, "message": error.message})
        raise CharError(error.message) from None
    except sdk_license.LicenseRateLimited:
        _record(license_id, last_error={"status": "rate_limited", "message": RATE_LIMITED})
        raise CharError(RATE_LIMITED) from None
    except sdk_license.LicenseUnreachable:
        _record(license_id, last_error={"status": "unreachable", "message": UNREACHABLE})
        raise CharError(UNREACHABLE) from None
    mark_verified(char, max(verdict.expires_at, time.monotonic() + RUN_GRACE_SECONDS))
    _record(license_id, verified_until=verdict.valid_until)

    def refused(status: str, message: str) -> None:
        _record(license_id, last_error={"status": status, "message": message or REPORT_REFUSED})

    if _first_report(license_id):
        sdk_license.report_usage(
            verdict.ticket,
            output,
            MODEL,
            CLIENT,
            base=sdk_license.api_base(secret),
            license_id=license_id,
            on_refused=refused,
        )
    return char


# --- routes, as plain functions so they test without aiohttp ----------------------------------

_BAD_BODY = {"error": "Send a license ID and a secret as JSON."}
_BAD_ID = {"error": "That is not a license ID."}
_BAD_SECRET = {"error": "That is not a license secret. Copy it again from the share email."}
_MISMATCH = {"error": "That secret and this license don't match: one is test, the other live."}
_BAD_CHAR = {"licensed": False, "error": "That character cannot be read."}


def post_license(body: bytes | None, content_type: str) -> tuple[int, dict[str, Any]]:
    # application/json only, so a cross-site form post cannot overwrite a stored secret.
    if body is None or content_type.split(";")[0].strip().lower() != "application/json":
        return 400, _BAD_BODY
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return 400, _BAD_BODY
    if not isinstance(data, dict) or set(data) != {"license_id", "secret"}:
        return 400, _BAD_BODY
    license_id, secret = data["license_id"], data["secret"]
    if not isinstance(license_id, str) or not LICENSE_ID.fullmatch(license_id):
        return 400, _BAD_ID
    if not isinstance(secret, str) or not SECRET.fullmatch(secret):
        return 400, _BAD_SECRET
    if license_id.split("_")[1] != secret.split("_")[1]:
        return 400, _MISMATCH
    store_secret(license_id, secret)
    return 200, {"stored": True}


def delete_license(license_id: str) -> tuple[int, dict[str, Any]]:
    if not LICENSE_ID.fullmatch(license_id):
        return 400, _BAD_ID
    forget_secret(license_id)
    return 200, {"stored": False}


def get_license(license_id: str) -> tuple[int, dict[str, Any]]:
    if not LICENSE_ID.fullmatch(license_id):
        return 400, _BAD_ID
    return 200, status_of(license_id)


def character_license(name: str, override: str = "") -> tuple[int, dict[str, Any]]:
    """Whether a character is licensed, resolved and confined exactly as Load Character does."""
    for value in (name, override):
        if len(value) > _MAX_NAME or "\x00" in value:
            return 400, _BAD_CHAR
    try:
        path = folders.resolve(name, override)
        char = open_cached(path)
        block = licensed(char)
        license_id = None
        if block is not None:
            if sdk_license is not None:
                found = sdk_license.read_license(char.manifest)
                license_id = found.license_id if found is not None else None
            elif isinstance(block, dict) and LICENSE_ID.fullmatch(
                str(block.get("license_id", ""))
            ):
                license_id = block["license_id"]
            else:
                return 400, _BAD_CHAR
    except (CharError, OSError, ValueError):
        return 400, _BAD_CHAR
    return 200, {
        "licensed": block is not None,
        "license_id": license_id,
        "name": label(name, override, path),
    }


async def _read_capped(request: Any) -> bytes | None:
    if request.content_length is not None and request.content_length > MAX_BODY:
        return None
    body = b""
    while True:
        chunk = await request.content.read(MAX_BODY + 1 - len(body))
        if not chunk:
            return body
        body += chunk
        if len(body) > MAX_BODY:
            return None


def register_routes() -> bool:
    """Add the routes to ComfyUI's server; False when there is none (tests, headless use)."""
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return False
    routes = PromptServer.instance.routes

    async def off_loop(work, *args):
        # The store and the .char are files; reading them must not stall the server's event loop.
        status, payload = await asyncio.get_running_loop().run_in_executor(None, work, *args)
        return web.json_response(payload, status=status)

    @routes.post("/omnichar/license")
    async def _post(request):
        body = await _read_capped(request)
        return await off_loop(post_license, body, request.headers.get("Content-Type", ""))

    @routes.delete("/omnichar/license/{license_id}")
    async def _delete(request):
        return await off_loop(delete_license, request.match_info.get("license_id", ""))

    @routes.get("/omnichar/license/{license_id}")
    async def _get(request):
        return await off_loop(get_license, request.match_info.get("license_id", ""))

    @routes.get("/omnichar/character-license")
    async def _character(request):
        query = request.rel_url.query
        name, override = query.get("char", ""), query.get("char_path", "")
        return await off_loop(character_license, name, override)

    return True
