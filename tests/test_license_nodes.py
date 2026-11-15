"""Licensed characters in the node pack, against a stub ComfyUI and a fake Omnichar."""

import asyncio
import importlib
import io
import json
import logging
import math
import os
import shutil
import socket
import stat
import sys
import threading
import types
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

pytest.importorskip("torch")

from comfystub import FakeClip, FakeModel, install  # noqa: E402
from conftest import FIXTURES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOUBLE = [f"transformer_blocks.0.attn.{p}" for p in ("to_q", "to_k", "to_v", "to_out.0")]

SECRET = "lsk_live_" + "Pr1v4teS3cret" * 3 + "abcd"
LICENSE_ID = "lic_live_Q2x8abcdefgh"
CHAR_ID = "7f1c2a9e-0000-4000-8000-000000000001"
TICKET = "lvt_" + "T1ck3t" * 7
BLOCK = {
    "v": 1,
    "license_id": LICENSE_ID,
    "issuer": "omnichar",
    "key": LICENSE_ID,
    "env": "production",
    "kid": "live-2026-11-09",
    "signature": "kR0abc",
    "char_id": CHAR_ID,
    "scope": {"use": "advertising", "media": ["image"]},
    "starts_at": "2026-12-01T00:00:00.000Z",
    "ends_at": "2027-06-01T00:00:00.000Z",
}
VALID = (
    200,
    {},
    json.dumps(
        {
            "status": "valid",
            "ticket": TICKET,
            "ttl_seconds": 600,
            "valid_until": "2099-01-01T00:00:00.000Z",
            "owner_name": "Omnicast Talents",
        }
    ).encode(),
)
ACCEPTED = (202, {}, b'{"accepted":true}')
REFUSALS = {
    "wrong_secret": "The license ID or secret is not right. Check the secret you were given.",
    "revoked": "This license was revoked by Omnicast Talents.",
    "expired": "This license expired on 12 March 2027.",
    "not_started": "This license starts on 1 December 2026.",
    "wrong_character": "This license is for a different character.",
    "output_not_allowed": "This license does not allow video.",
    "limit_reached": "This license has used all of its generations. Ask Omnicast Talents for more.",
}

#: Captured from the pack before licensing existed (HEAD 251c02e), for Ada.char unlicensed.
BASELINE_PROMPT = {
    "ordinal": "Images 1 and 2 show Ada, the same character in every image. Ada, a woman in her "
    "thirties with short dark hair and a scar through her left eyebrow. in the rain",
    "token": "<Picture 1> <Picture 2> show Ada, the same character in every image. Ada, a woman "
    "in her thirties with short dark hair and a scar through her left eyebrow. in the rain",
}
BASELINE_IMAGES = {"ordinal": "0ea3667eabc41271", "token": "0ea3667eabc41271"}
BASELINE_SHEET = "6bd60be9db1f0a98"
BASELINE_PICK = ("c8b5fc75b40aa8a1", "body", 2)
BASELINE_SPLIT = ["1ef26d78ce4aa4ef", "c8b5fc75b40aa8a1", None, None, None]


def licensed_copy(source: Path, target: Path, block=BLOCK) -> Path:
    """A copy of a .char whose manifest carries ``reserved.license``, every other byte the same."""
    with zipfile.ZipFile(source) as archive:
        members = [(info, archive.read(info.filename)) for info in archive.infolist()]
    with zipfile.ZipFile(target, "w") as out:
        for info, data in members:
            if info.filename == "manifest.json":
                manifest = json.loads(data)
                manifest.setdefault("reserved", {})["license"] = block
                data = json.dumps(manifest, separators=(",", ":")).encode()
            out.writestr(info, data, compress_type=info.compress_type)
    return target


class FakeOmnichar:
    """Stands in for license._post: scripted verify answers, usage always accepted."""

    def __init__(self, *verify_answers):
        self.answers = list(verify_answers) or [VALID]
        self.usage = ACCEPTED
        self.calls = []

    def __call__(self, url, payload, timeout):
        self.calls.append((url, payload))
        if url.endswith("/licenses/usage"):
            return self.usage
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def count(self, route):
        return sum(1 for url, _ in self.calls if url.endswith(route))


def _load_pack(tmp_path, monkeypatch, old_sdk=False):
    models = tmp_path / "models"
    (models / "characters").mkdir(parents=True)
    (models / "loras").mkdir(parents=True)
    shutil.copy(FIXTURES / "ada.char", models / "characters" / "Ada.char")
    licensed_copy(FIXTURES / "ada.char", models / "characters" / "Lic.char")
    install(models, unet_stems=DOUBLE, user_dir=tmp_path / "user")
    monkeypatch.syspath_prepend(str(ROOT.parent))
    saved = {n: m for n, m in sys.modules.items() if n.startswith("omnichar_sdk")}
    for name in list(saved):
        del sys.modules[name]
    for name in [m for m in list(sys.modules) if m.startswith("omnichar_pack")]:
        del sys.modules[name]
    sys.path.insert(0, str(ROOT / "packages/omnichar-sdk/src"))
    sdk = importlib.import_module("omnichar_sdk")
    if old_sdk:
        # An SDK from before 0.1.2: the submodule does not exist and cannot be imported.
        del sdk.license
        sys.modules["omnichar_sdk.license"] = None  # type: ignore[assignment]
    spec = importlib.util.spec_from_file_location(
        "omnichar_pack", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["omnichar_pack"] = module
    spec.loader.exec_module(module)
    return module, models, saved


@pytest.fixture
def pack(tmp_path, monkeypatch):
    module, models, saved = _load_pack(tmp_path, monkeypatch)
    licensing = sys.modules["omnichar_pack.nodes.licensing"]
    sdk = licensing.sdk_license
    sdk.clear_cache()
    monkeypatch.delenv(sdk.ENV_OVERRIDE, raising=False)
    monkeypatch.setattr(sdk, "_sleep", lambda seconds: None)
    server = FakeOmnichar()
    monkeypatch.setattr(sdk, "_post", server)
    threads = []
    report = sdk.report_usage
    monkeypatch.setattr(
        sdk, "report_usage", lambda *a, **k: threads.append(report(*a, **k)) or threads[-1]
    )
    env = types.SimpleNamespace(
        module=module,
        nodes=module.NODE_CLASS_MAPPINGS,
        models=models,
        user=tmp_path / "user",
        licensing=licensing,
        sdk=sdk,
        server=server,
        settle=lambda: [t.join(5) for t in threads],
    )
    yield env
    sdk.clear_cache()
    sys.modules.update(saved)


@pytest.fixture
def old_pack(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="omnichar")
    module, models, saved = _load_pack(tmp_path, monkeypatch, old_sdk=True)
    yield module, models, sys.modules["omnichar_pack.nodes.licensing"]
    sys.modules.pop("omnichar_sdk.license", None)
    sys.modules.update(saved)


def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a socket was opened")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def queue(env, name="Lic.char", output="image", style="ordinal"):
    """One queued run as ComfyUI drives it: validate, IS_CHANGED, then load and decode."""
    loader = env.nodes["OmnicharLoadCharacter"]
    verdict = loader.VALIDATE_INPUTS(name, "")
    assert verdict is True, verdict
    stamp = loader.IS_CHANGED(name, "", output)
    (char,) = loader().load(name, "", output)
    decoded = env.nodes["OmnicharDecodeCharacter"]().decode(char, style, None, "in the rain")
    env.settle()
    return stamp, char, decoded


def digest(tensor):
    import hashlib

    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()[:16]


# --- unlicensed: unchanged and offline --------------------------------------------------------


@pytest.mark.parametrize("style", ["ordinal", "token"])
def test_an_unlicensed_character_runs_as_before_and_never_goes_online(pack, monkeypatch, style):
    no_network(monkeypatch)
    loader = pack.nodes["OmnicharLoadCharacter"]
    target = pack.models / "characters" / "Ada.char"
    status = os.stat(target)

    assert loader.VALIDATE_INPUTS("Ada.char") is True
    assert loader.VALIDATE_INPUTS("Ada.char", "") is True
    assert loader.IS_CHANGED("Ada.char") == (status.st_mtime_ns, status.st_size)
    assert loader.IS_CHANGED("Ada.char", "", "video") == (status.st_mtime_ns, status.st_size)

    (char,) = loader().load("Ada.char")
    cond, images, refs, sheet, prompt, voice = pack.nodes["OmnicharDecodeCharacter"]().decode(
        char, style, FakeClip(), "in the rain"
    )
    assert prompt == BASELINE_PROMPT[style]
    assert digest(images) == BASELINE_IMAGES[style]
    assert digest(sheet) == BASELINE_SHEET and voice is None
    image, role, count = pack.nodes["OmnicharCharacterReference"]().pick(refs, 1)
    assert (digest(image), role, count) == BASELINE_PICK
    split = pack.nodes["OmnicharCharacterReferencesSplit"]().split(refs)
    assert [digest(x) if x is not None else None for x in split[:-1]] == BASELINE_SPLIT
    model, _ = pack.nodes["OmnicharApplyCharacterLoRA"]().apply(FakeModel(DOUBLE), char, -1.0)
    assert model[2] == pytest.approx(0.8)
    saved = pack.nodes["OmnicharSaveCharacter"]().save(char, "copy")
    assert Path(saved["result"][0]).read_bytes() == target.read_bytes()
    assert pack.server.calls == []
    assert not (pack.user / "omnichar").exists()


def test_old_validate_inputs_signature_is_kept(pack):
    import inspect

    # ComfyUI skips its own combo check for any input VALIDATE_INPUTS names, so output stays out.
    loader = pack.nodes["OmnicharLoadCharacter"]
    assert list(inspect.signature(loader.VALIDATE_INPUTS).parameters) == ["char", "char_path"]


# --- the output widget and saved workflows ----------------------------------------------------


def _widgets(cls):
    spec = cls.INPUT_TYPES()
    names = []
    for group in ("required", "optional"):
        for name, definition in spec.get(group, {}).items():
            kind, options = definition[0], (definition[1] if len(definition) > 1 else {})
            if options.get("forceInput"):
                continue
            if isinstance(kind, list) or kind in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                names.append(name)
    return names


def test_the_output_widget_comes_last_so_saved_values_keep_their_places(pack):
    loader = pack.nodes["OmnicharLoadCharacter"]
    assert _widgets(loader) == ["char", "char_path", "output"]
    kind, options = loader.INPUT_TYPES()["optional"]["output"]
    assert kind == ["image", "video"]
    assert options["default"] == "image" and options["advanced"] is True
    assert options["tooltip"] == "What you'll generate. Only used for licensed characters."
    assert list(loader.INPUT_TYPES()["required"]) == ["char"]


def test_every_shipped_workflow_maps_its_values_to_the_same_inputs(pack):
    """Before licensing, Load Character had two widgets; a saved ['sia.char', ''] still means
    char and char_path, and output falls back to its default."""
    before = {"OmnicharLoadCharacter": ["char", "char_path"]}
    paths = sorted((ROOT / "workflows").rglob("*.json"))
    assert any(p.parent.name == "voice" for p in paths)
    for path in paths:
        wf = json.loads(path.read_text())
        for node in wf["nodes"]:
            if not node["type"].startswith("Omnichar"):
                continue
            cls = pack.nodes.get(node["type"])
            assert cls is not None, f"{path.name}: {node['type']}"
            names = _widgets(cls)
            values = node.get("widgets_values", [])
            assert len(values) <= len(names), f"{path.name}: {node['type']}"
            if node["type"] in before:
                assert names[: len(before[node["type"]])] == before[node["type"]]
                mapped = dict(zip(names, values, strict=False))
                assert set(mapped) == {"char", "char_path"}, path.name
                assert "output" not in mapped


def test_the_licensed_workflow_differs_from_flux_only_in_the_character_and_its_note(pack):
    flux = json.loads((ROOT / "workflows/flux_klein_9b_image_char.json").read_text())
    ours = json.loads((ROOT / "workflows/flux_klein_9b_image_licensed_char.json").read_text())
    notes = [n for n in ours["nodes"] if n["type"] == "Note"]
    assert len(notes) == 1
    assert notes[0]["widgets_values"] == [
        "Licensed character\n"
        "1. Put your licensed .char in models/characters and pick it here.\n"
        "2. Click Enter secret on this node and paste the license secret from your share email.\n"
        "3. Queue. OmniChar checks the license before every run; if it's revoked or expired, "
        "the run stops with the reason.\n"
        "Unlicensed characters work as usual and never go online."
    ]
    assert notes[0]["id"] == flux["last_node_id"] + 1 == ours["last_node_id"]
    rest = [n for n in ours["nodes"] if n["type"] != "Note"]
    for node in rest:
        if node["type"] == "OmnicharLoadCharacter":
            assert node["widgets_values"] == ["licensed-example.char", ""]
            node["widgets_values"] = ["sia.char", ""]
    assert rest == flux["nodes"]
    for key in ("links", "groups", "config", "extra", "version", "last_link_id"):
        assert ours[key] == flux[key], key
    assert "lsk_" not in json.dumps(ours)


def test_the_client_version_is_the_pack_version(pack):
    text = (ROOT / "pyproject.toml").read_text()
    assert f'version = "{pack.licensing.PACK_VERSION}"' in text


# --- a licensed character ---------------------------------------------------------------------


def test_no_secret_stops_the_queue_offline(pack, monkeypatch):
    no_network(monkeypatch)
    loader = pack.nodes["OmnicharLoadCharacter"]
    message = "Lic.char is licensed. Enter the license secret from the share email."
    assert loader.VALIDATE_INPUTS("Lic.char") == message
    from omnichar_sdk import CharError

    with pytest.raises(CharError, match="Enter the license secret"):
        loader().load("Lic.char")
    assert pack.server.calls == []


def test_a_verified_character_runs_and_reports_once(pack):
    pack.licensing.store_secret(LICENSE_ID, SECRET)
    stamp, char, decoded = queue(pack)
    assert math.isnan(stamp)
    assert decoded[4] == BASELINE_PROMPT["ordinal"]
    assert pack.server.count("/licenses/verify") == 1
    assert pack.server.count("/licenses/usage") == 1
    usage = [p for u, p in pack.server.calls if u.endswith("/usage")][0]
    assert usage == {
        "ticket": TICKET, "output": "image", "model": "unspecified", "client": "comfyui"
    }
    verify = [p for u, p in pack.server.calls if u.endswith("/verify")][0]
    assert verify["char_id"] == CHAR_ID and verify["license_id"] == LICENSE_ID
    assert verify["client"] == "comfyui" and verify["model"] == "unspecified"

    # Downstream nodes in the same run report nothing more.
    pack.nodes["OmnicharApplyCharacterLoRA"]().apply(FakeModel(DOUBLE), char, -1.0)
    pack.nodes["OmnicharDecodeCharacter"]().decode(char, "token")
    pack.settle()
    assert pack.server.count("/licenses/usage") == 1

    # A second queued run: verified from the cache, reported again.
    queue(pack, output="image")
    assert pack.server.count("/licenses/verify") == 1
    assert pack.server.count("/licenses/usage") == 2
    status = pack.licensing.status_of(LICENSE_ID)
    assert status == {"stored": True, "verified_until": "2099-01-01T00:00:00.000Z"}


def test_the_output_reaches_verify(pack):
    pack.licensing.store_secret(LICENSE_ID, SECRET)
    queue(pack, output="video")
    verify = [p for u, p in pack.server.calls if u.endswith("/verify")][0]
    assert verify["output"] == "video"


def test_a_failing_usage_report_does_not_fail_the_run(pack, monkeypatch):
    pack.licensing.store_secret(LICENSE_ID, SECRET)

    def flaky(url, payload, timeout):
        if url.endswith("/usage"):
            raise OSError("down")
        return VALID

    monkeypatch.setattr(pack.sdk, "_post", flaky)
    _stamp, _char, decoded = queue(pack)
    assert decoded[4] == BASELINE_PROMPT["ordinal"]


@pytest.mark.parametrize("status", sorted(REFUSALS))
def test_a_refusal_stops_the_run_with_the_servers_message(pack, status):
    from omnichar_sdk import CharError

    pack.licensing.store_secret(LICENSE_ID, SECRET)
    pack.server.answers = [
        (403, {}, json.dumps({"status": status, "message": REFUSALS[status]}).encode())
    ]
    with pytest.raises(CharError) as excinfo:
        pack.nodes["OmnicharLoadCharacter"]().load("Lic.char")
    assert str(excinfo.value) == REFUSALS[status]
    assert pack.server.count("/licenses/usage") == 0
    assert pack.licensing.status_of(LICENSE_ID)["last_error"] == {
        "status": status,
        "message": REFUSALS[status],
    }


def test_too_many_attempts_is_one_call_and_a_plain_message(pack):
    from omnichar_sdk import CharError

    pack.licensing.store_secret(LICENSE_ID, SECRET)
    pack.server.answers = [(429, {"Retry-After": "900"}, b"{}")]
    with pytest.raises(CharError) as excinfo:
        pack.nodes["OmnicharLoadCharacter"]().load("Lic.char")
    assert str(excinfo.value) == "Too many attempts. Try again in a few minutes."
    assert len(pack.server.calls) == 1


def test_no_network_is_unreachable(pack):
    import urllib.error

    from omnichar_sdk import CharError

    pack.licensing.store_secret(LICENSE_ID, SECRET)
    pack.server.answers = [urllib.error.URLError("offline")]
    with pytest.raises(CharError) as excinfo:
        pack.nodes["OmnicharLoadCharacter"]().load("Lic.char")
    assert str(excinfo.value) == (
        "Can't reach OmniChar to check this license. Connect and try again."
    )


def test_a_cached_verdict_carries_a_run_through_a_network_drop(pack):
    import urllib.error

    pack.licensing.store_secret(LICENSE_ID, SECRET)
    queue(pack)
    pack.server.answers = [urllib.error.URLError("offline")]
    queue(pack)
    assert pack.server.count("/licenses/verify") == 1


def test_licensed_is_never_cached_by_comfyui(pack):
    loader = pack.nodes["OmnicharLoadCharacter"]
    assert math.isnan(loader.IS_CHANGED("Lic.char"))
    assert math.isnan(loader.IS_CHANGED("Lic.char", "", "video"))


def test_downstream_nodes_refuse_an_unverified_licensed_character(pack):
    from omnichar_sdk import Character, CharError

    char = Character.open(pack.models / "characters" / "Lic.char")
    message = "This licensed character hasn't been verified. Re-queue to check its license."
    for call in (
        lambda: pack.nodes["OmnicharDecodeCharacter"]().decode(char, "ordinal"),
        lambda: pack.nodes["OmnicharApplyCharacterLoRA"]().apply(FakeModel(DOUBLE), char, -1.0),
        lambda: pack.nodes["OmnicharSaveCharacter"]().save(char, "leak"),
    ):
        with pytest.raises(CharError) as excinfo:
            call()
        assert str(excinfo.value) == message
    assert not (pack.models / "characters" / "leak.char").exists()


def test_a_cleared_character_lasts_the_run_even_past_its_ticket(pack, monkeypatch):
    """A slow model load must not fail a run whose license was checked at its start."""
    import time

    from omnichar_sdk import CharError

    pack.licensing.store_secret(LICENSE_ID, SECRET)
    pack.server.answers = [
        (200, {}, json.dumps({"status": "valid", "ticket": TICKET, "ttl_seconds": 600,
                              "valid_until": "2099-01-01T00:00:00.000Z"}).encode())
    ]
    started = time.monotonic()
    _stamp, char, _decoded = queue(pack)
    common = sys.modules["omnichar_pack.nodes.common"]
    deadline = common._verified[char]
    assert deadline >= started + pack.licensing.RUN_GRACE_SECONDS

    monkeypatch.setattr(common, "time", types.SimpleNamespace(monotonic=lambda: started + 900))
    pack.nodes["OmnicharDecodeCharacter"]().decode(char, "ordinal")  # ticket gone, run goes on

    monkeypatch.setattr(common, "time", types.SimpleNamespace(monotonic=lambda: deadline + 1))
    with pytest.raises(CharError, match="hasn't been verified"):
        pack.nodes["OmnicharDecodeCharacter"]().decode(char, "ordinal")


def test_a_refused_usage_report_shows_and_the_next_run_checks_again(pack):
    """max_generations: the report that uses the last one is refused; the next run is stopped."""
    from omnichar_sdk import CharError

    message = REFUSALS["limit_reached"]
    refusal = (403, {}, json.dumps({"status": "limit_reached", "message": message}).encode())
    pack.licensing.store_secret(LICENSE_ID, SECRET)
    pack.server.usage = refusal
    queue(pack)
    status = pack.licensing.status_of(LICENSE_ID)
    assert status["last_error"] == {"status": "limit_reached", "message": message}

    pack.server.answers = [refusal]
    with pytest.raises(CharError) as excinfo:
        pack.nodes["OmnicharLoadCharacter"]().load("Lic.char")
    assert str(excinfo.value) == message
    assert pack.server.count("/licenses/verify") == 2


def _run(monkeypatch, prompt_id):
    utils = types.ModuleType("comfy_execution.utils")
    utils.get_executing_context = lambda: types.SimpleNamespace(prompt_id=prompt_id)
    package = types.ModuleType("comfy_execution")
    package.utils = utils
    monkeypatch.setitem(sys.modules, "comfy_execution", package)
    monkeypatch.setitem(sys.modules, "comfy_execution.utils", utils)


def test_two_loaders_on_one_character_report_once_per_run(pack, monkeypatch):
    pack.licensing.store_secret(LICENSE_ID, SECRET)
    loader = pack.nodes["OmnicharLoadCharacter"]
    _run(monkeypatch, "prompt-1")
    loader().load("Lic.char", "", "image")
    loader().load("Lic.char", "", "image")
    pack.settle()
    assert pack.server.count("/licenses/usage") == 1
    _run(monkeypatch, "prompt-2")
    loader().load("Lic.char", "", "image")
    pack.settle()
    assert pack.server.count("/licenses/usage") == 2


def test_save_keeps_the_license_exactly_and_encode_never_adds_one(pack):
    import torch
    from omnichar_sdk import Character

    pack.licensing.store_secret(LICENSE_ID, SECRET)
    _stamp, char, _decoded = queue(pack)
    saved = pack.nodes["OmnicharSaveCharacter"]().save(char, "kept")
    source = pack.models / "characters" / "Lic.char"
    assert Path(saved["result"][0]).read_bytes() == source.read_bytes()
    assert Character.open(saved["result"][0]).manifest.reserved["license"] == BLOCK

    (built,) = pack.nodes["OmnicharEncodeCharacter"]().encode(
        "Bo", "A tall man.", 512, face=torch.rand(1, 96, 64, 3)
    )
    assert "license" not in built.manifest.reserved


def test_a_damaged_license_block_is_refused_not_run(pack):
    from omnichar_sdk import CharError

    licensed_copy(FIXTURES / "ada.char", pack.models / "characters" / "Bad.char", ["junk"])
    loader = pack.nodes["OmnicharLoadCharacter"]
    assert "license block is unreadable" in loader.VALIDATE_INPUTS("Bad.char")
    with pytest.raises(CharError, match="unreadable"):
        loader().load("Bad.char")
    assert pack.server.calls == []


# --- an old omnichar-sdk ----------------------------------------------------------------------


def test_an_old_sdk_still_loads_and_runs_unlicensed_characters(old_pack, caplog, monkeypatch):
    module, _models, licensing = old_pack
    no_network(monkeypatch)
    assert licensing.sdk_license is None
    assert len(module.NODE_CLASS_MAPPINGS) == 8
    lines = [r.getMessage() for r in caplog.get_records("setup") if "too old" in r.getMessage()]
    assert len(lines) == 1 and "pip install -U 'omnichar-sdk>=0.1.2,<0.2'" in lines[0]
    (char,) = module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"]().load("Ada.char")
    prompt = module.NODE_CLASS_MAPPINGS["OmnicharDecodeCharacter"]().decode(char, "ordinal")[4]
    assert prompt.startswith("Images 1 and 2 show Ada,")


def test_an_old_sdk_refuses_a_licensed_character(old_pack):
    from omnichar_sdk import CharError

    module, _models, licensing = old_pack
    loader = module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"]
    assert loader.VALIDATE_INPUTS("Lic.char") == "Update omnichar-sdk to use licensed characters."
    with pytest.raises(CharError, match="Update omnichar-sdk"):
        loader().load("Lic.char")
    assert licensing.character_license("Lic.char") == (
        200,
        {"licensed": True, "license_id": LICENSE_ID, "name": "Lic"},
    )


# --- routes -----------------------------------------------------------------------------------


class Content:
    def __init__(self, body):
        self.stream = io.BytesIO(body)

    async def read(self, n=-1):
        return self.stream.read(n)


class Request:
    def __init__(self, body=b"", content_type="application/json", match=None, query=None,
                 length="auto"):
        self.content = Content(body)
        self.content_length = len(body) if length == "auto" else length
        self.headers = {"Content-Type": content_type}
        self.match_info = match or {}
        self.rel_url = types.SimpleNamespace(query=query or {})


def call(pack, method, path, request):
    from server import PromptServer

    handler = PromptServer.instance.routes.handlers[(method, path)]
    response = asyncio.run(handler(request))
    return response.status, json.loads(response.body)


def post(pack, payload, **kwargs):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return call(pack, "POST", "/omnichar/license", Request(body, **kwargs))


def test_the_routes_are_registered(pack):
    from server import PromptServer

    assert set(PromptServer.instance.routes.handlers) == {
        ("POST", "/omnichar/license"),
        ("DELETE", "/omnichar/license/{license_id}"),
        ("GET", "/omnichar/license/{license_id}"),
        ("GET", "/omnichar/character-license"),
    }


def test_storing_a_secret_is_private_and_never_echoed(pack):
    status, body = post(pack, {"license_id": LICENSE_ID, "secret": SECRET})
    assert (status, body) == (200, {"stored": True})
    store = pack.user / "omnichar" / "licenses.json"
    assert stat.S_IMODE(store.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.parent.stat().st_mode) == 0o700
    assert json.loads(store.read_text()) == {LICENSE_ID: SECRET}

    status, body = call(
        pack, "GET", "/omnichar/license/{license_id}", Request(match={"license_id": LICENSE_ID})
    )
    assert (status, body) == (200, {"stored": True})

    status, body = call(
        pack, "DELETE", "/omnichar/license/{license_id}", Request(match={"license_id": LICENSE_ID})
    )
    assert (status, body) == (200, {"stored": False})
    assert json.loads(store.read_text()) == {}


def test_storing_a_secret_drops_cached_verdicts_and_old_errors(pack):
    pack.licensing.store_secret(LICENSE_ID, SECRET)
    queue(pack)
    post(pack, {"license_id": LICENSE_ID, "secret": SECRET})
    assert pack.licensing.status_of(LICENSE_ID) == {"stored": True}
    queue(pack)
    assert pack.server.count("/licenses/verify") == 2


@pytest.mark.parametrize(
    ("payload", "kwargs"),
    [
        (b"not json", {}),
        (b"[]", {}),
        ({"license_id": LICENSE_ID}, {}),
        ({"license_id": LICENSE_ID, "secret": SECRET, "extra": 1}, {}),
        ({"license_id": "lic_live_short", "secret": SECRET}, {}),
        ({"license_id": "../../etc/passwd", "secret": SECRET}, {}),
        ({"license_id": LICENSE_ID, "secret": "lsk_live_short"}, {}),
        ({"license_id": LICENSE_ID, "secret": "sk-" + "a" * 40}, {}),
        ({"license_id": LICENSE_ID, "secret": "lsk_test_" + "a" * 43}, {}),
        ({"license_id": LICENSE_ID, "secret": SECRET}, {"content_type": "text/plain"}),
        ({"license_id": LICENSE_ID, "secret": SECRET + "x" * 5000}, {}),
        ({"license_id": LICENSE_ID, "secret": SECRET}, {"length": 99999}),
    ],
)
def test_bad_posts_get_a_fixed_400_without_the_secret(pack, payload, kwargs):
    status, body = post(pack, payload, **kwargs)
    assert status == 400
    assert set(body) == {"error"}
    assert SECRET not in json.dumps(body) and "lsk_" not in json.dumps(body)
    assert not (pack.user / "omnichar" / "licenses.json").exists()


def test_a_chunked_body_over_the_cap_is_refused(pack):
    body = json.dumps({"license_id": LICENSE_ID, "secret": SECRET, "pad": "x" * 5000}).encode()
    assert post(pack, body, length=None)[0] == 400


@pytest.mark.parametrize("license_id", ["", "lic_live_x", "..", "lic_live_" + "a" * 61, "abc"])
def test_bad_license_ids_are_refused_on_every_route(pack, license_id):
    request = Request(match={"license_id": license_id})
    assert call(pack, "GET", "/omnichar/license/{license_id}", request)[0] == 400
    assert call(pack, "DELETE", "/omnichar/license/{license_id}", request)[0] == 400


def test_a_failed_write_leaves_the_old_store_intact(pack, monkeypatch):
    pack.licensing.store_secret(LICENSE_ID, SECRET)
    store = pack.user / "omnichar" / "licenses.json"
    before = store.read_bytes()

    def broken(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(pack.licensing.os, "replace", broken)
    with pytest.raises(OSError):
        pack.licensing.store_secret("lic_live_otherlicense1", "lsk_live_" + "Q" * 43)
    assert store.read_bytes() == before
    assert sorted(p.name for p in store.parent.iterdir()) == ["licenses.json"]


def test_character_license_reports_what_load_character_would_open(pack):
    route = "/omnichar/character-license"
    assert call(pack, "GET", route, Request(query={"char": "Ada.char"})) == (
        200,
        {"licensed": False, "license_id": None, "name": "Ada"},
    )
    assert call(pack, "GET", route, Request(query={"char": "Lic.char"})) == (
        200,
        {"licensed": True, "license_id": LICENSE_ID, "name": "Lic"},
    )
    inside = str(pack.models / "characters" / "Lic.char")
    assert call(pack, "GET", route, Request(query={"char": "", "char_path": inside}))[1][
        "licensed"
    ]


@pytest.mark.parametrize(
    "query",
    [
        {"char": "../../../../etc/passwd"},
        {"char": "/etc/passwd"},
        {"char": "../loras/x.char"},
        {"char": "missing.char"},
        {"char": ""},
        {"char": "Ada.char\x00"},
        {"char": "x" * 300},
        {"char": "", "char_path": "/etc/passwd"},
        {"char": "", "char_path": "~/.ssh/id_rsa"},
    ],
)
def test_character_license_refuses_anything_outside_the_character_folders(pack, query):
    status, body = call(pack, "GET", "/omnichar/character-license", Request(query=query))
    assert status == 400
    assert body["licensed"] is False
    assert "/" not in body["error"] and str(pack.models) not in json.dumps(body)


def test_a_symlinked_character_is_seen_exactly_as_load_character_sees_it(pack, tmp_path):
    """Load Character opens a listed name that is a symlink, so the buttons must appear for it."""
    outside = licensed_copy(FIXTURES / "ada.char", tmp_path / "elsewhere.char")
    (pack.models / "characters" / "link.char").symlink_to(outside)
    status, body = call(
        pack, "GET", "/omnichar/character-license", Request(query={"char": "link.char"})
    )
    assert (status, body) == (200, {"licensed": True, "license_id": LICENSE_ID, "name": "link"})
    loader = pack.nodes["OmnicharLoadCharacter"]
    assert loader.VALIDATE_INPUTS("link.char") == (
        "link.char is licensed. Enter the license secret from the share email."
    )
    # char_path keeps Load's confinement: the target itself is outside the character folders.
    status, _ = call(
        pack, "GET", "/omnichar/character-license",
        Request(query={"char": "", "char_path": str(outside)}),
    )
    assert status == 400


def test_a_trailing_newline_is_not_a_license_id_or_secret(pack):
    assert post(pack, {"license_id": LICENSE_ID + "\n", "secret": SECRET})[0] == 400
    assert post(pack, {"license_id": LICENSE_ID, "secret": SECRET + "\n"})[0] == 400
    request = Request(match={"license_id": LICENSE_ID + "\n"})
    assert call(pack, "GET", "/omnichar/license/{license_id}", request)[0] == 400


# --- the secret goes nowhere else -------------------------------------------------------------


def test_the_secret_appears_nowhere_but_the_store(pack, monkeypatch, caplog, tmp_path):
    import urllib.error

    from omnichar_sdk import CharError
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    caplog.set_level(logging.DEBUG)
    seen = []
    seen.append(json.dumps(post(pack, {"license_id": LICENSE_ID, "secret": SECRET})))
    seen.append(
        json.dumps(
            call(
                pack,
                "GET",
                "/omnichar/license/{license_id}",
                Request(match={"license_id": LICENSE_ID}),
            )
        )
    )
    _stamp, char, decoded = queue(pack)
    saved = pack.nodes["OmnicharSaveCharacter"]().save(char, "out")
    seen += [repr(char), repr(decoded[4]), json.dumps(saved["ui"])]

    for answer in [
        (403, {}, json.dumps({"status": "revoked", "message": REFUSALS["revoked"]}).encode()),
        (429, {}, b""),
        urllib.error.URLError(f"echo {SECRET}"),
        (500, {}, SECRET.encode()),
    ]:
        pack.sdk.clear_cache()
        pack.server.answers = [answer]
        with pytest.raises(CharError) as excinfo:
            pack.nodes["OmnicharLoadCharacter"]().load("Lic.char")
        error = excinfo.value
        seen += [str(error), repr(error), repr(error.__cause__), repr(error.__context__)]
    seen.append(json.dumps(pack.licensing.status_of(LICENSE_ID)))

    # What ComfyUI saves: the node's widget values in the workflow and prompt, then the PNG.
    node = {"id": 1, "type": "OmnicharLoadCharacter", "widgets_values": ["Lic.char", "", "image"]}
    prompt = {"1": {"class_type": "OmnicharLoadCharacter",
                    "inputs": {"char": "Lic.char", "char_path": "", "output": "image"}}}
    info = PngInfo()
    info.add_text("prompt", json.dumps(prompt))
    info.add_text("workflow", json.dumps({"nodes": [node]}))
    png = io.BytesIO()
    Image.new("RGB", (4, 4)).save(png, "PNG", pnginfo=info)

    text = "\n".join(seen) + caplog.text
    assert SECRET not in text
    assert SECRET.encode() not in png.getvalue()
    store = pack.user / "omnichar" / "licenses.json"
    for path in tmp_path.rglob("*"):
        if path.is_file() and path != store:
            assert SECRET.encode() not in path.read_bytes(), path


# --- end to end, against a local Omnichar on a real socket ------------------------------------


def test_end_to_end_against_a_local_server(pack, monkeypatch):
    """The real _post over HTTP, pointed at this machine through OMNICHAR_LICENSE_API."""
    from omnichar_sdk import CharError

    monkeypatch.undo()  # drop the fake _post; the real one talks to the server below
    monkeypatch.setattr(pack.sdk, "_sleep", lambda seconds: None)
    script = []
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, payload))
            status, body = (202, {"accepted": True}) if self.path.endswith("usage") else script[0]
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            if status == 429:
                self.send_header("Retry-After", "60")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_port
    monkeypatch.setenv(pack.sdk.ENV_OVERRIDE, f"http://127.0.0.1:{port}/v1")
    pack.licensing.store_secret(LICENSE_ID, SECRET)
    loader = pack.nodes["OmnicharLoadCharacter"]
    threads = []
    report = pack.sdk.report_usage
    monkeypatch.setattr(
        pack.sdk, "report_usage", lambda *a, **k: threads.append(report(*a, **k)) or threads[-1]
    )
    try:
        script[:] = [json.loads(VALID[2])]
        script[0] = (200, script[0])
        (char,) = loader().load("Lic.char")
        pack.nodes["OmnicharDecodeCharacter"]().decode(char, "ordinal")
        [t.join(5) for t in threads]
        assert [p for p, _ in received] == ["/v1/licenses/verify", "/v1/licenses/usage"]
        assert received[0][1]["secret"] == SECRET

        for status, message in REFUSALS.items():
            pack.sdk.clear_cache()
            script[:] = [(403, {"status": status, "message": message})]
            with pytest.raises(CharError) as excinfo:
                loader().load("Lic.char")
            assert str(excinfo.value) == message

        pack.sdk.clear_cache()
        script[:] = [(429, {"error": {"code": "rate_limited", "message": "x"}})]
        with pytest.raises(CharError, match="^Too many attempts. Try again in a few minutes.$"):
            loader().load("Lic.char")
    finally:
        server.shutdown()
        server.server_close()

    pack.sdk.clear_cache()
    with pytest.raises(CharError, match="^Can't reach OmniChar to check this license"):
        loader().load("Lic.char")
