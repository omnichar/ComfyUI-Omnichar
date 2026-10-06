"""The ComfyUI node pack, against a stub of the parts of ComfyUI it touches."""

import shutil
import sys
from pathlib import Path

import pytest
from comfystub import FakeClip, FakeModel, install
from conftest import FIXTURES

ROOT = Path(__file__).resolve().parents[1]

DOUBLE = [f"transformer_blocks.0.attn.{p}" for p in ("to_q", "to_k", "to_v", "to_out.0")]


@pytest.fixture
def pack(tmp_path, monkeypatch):
    """The pack, loaded fresh against a stub ComfyUI whose models dir is empty but real."""
    models = tmp_path / "models"
    (models / "characters").mkdir(parents=True)
    (models / "loras").mkdir(parents=True)
    shutil.copy(FIXTURES / "ada.char", models / "characters" / "Ada.char")
    install(models, unet_stems=DOUBLE)
    monkeypatch.syspath_prepend(str(ROOT.parent))
    # Restored afterwards: dropping omnichar_sdk permanently would leave later tests patching a
    # stale module object, and a size cap that silently stops applying is worse than a failure.
    saved = {n: m for n, m in sys.modules.items() if n.startswith("omnichar_sdk")}
    for name in list(saved):
        del sys.modules[name]
    sys.path.insert(0, str(ROOT / "packages/omnichar-sdk/src"))
    for name in [m for m in list(sys.modules) if "nodes." in m or m.endswith(".nodes")]:
        del sys.modules[name]
    import importlib

    spec = importlib.util.spec_from_file_location(
        "omnichar_pack", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["omnichar_pack"] = module
    spec.loader.exec_module(module)
    yield module, models
    sys.modules.update(saved)


def test_every_node_registers_with_a_display_name(pack):
    module, _ = pack
    assert len(module.NODE_CLASS_MAPPINGS) == 8
    assert set(module.NODE_CLASS_MAPPINGS) == set(module.NODE_DISPLAY_NAME_MAPPINGS)


def test_every_character_input_is_forced_to_a_socket(pack):
    module, _ = pack
    for name, cls in module.NODE_CLASS_MAPPINGS.items():
        spec = cls.INPUT_TYPES()
        for group in ("required", "optional"):
            for field, definition in spec.get(group, {}).items():
                if definition[0] == "CHARACTER":
                    # Without forceInput the frontend draws a widget for a type it does not know.
                    assert definition[1].get("forceInput") is True, f"{name}.{field}"


def test_the_loader_lists_and_opens_a_character(pack):
    module, _ = pack
    loader = module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"]()
    assert "Ada.char" in module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"].INPUT_TYPES()[
        "required"
    ]["char"][0]
    (char,) = loader.load("Ada.char")
    assert char.name == "Ada"


def test_is_changed_is_a_stat_not_a_hash(pack):
    module, models = pack
    cls = module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"]
    stamp = cls.IS_CHANGED("Ada.char")
    assert isinstance(stamp, tuple) and len(stamp) == 2
    target = models / "characters" / "Ada.char"
    target.touch()
    assert cls.IS_CHANGED("Ada.char") != stamp


@pytest.mark.parametrize(
    "override",
    ["/etc/passwd", "../../../../etc/shadow", "~/.ssh/id_rsa"],
)
def test_char_path_cannot_read_outside_the_character_directories(pack, override):
    module, _ = pack
    from omnichar_sdk import CharError

    loader = module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"]()
    with pytest.raises(CharError) as excinfo:
        loader.load("Ada.char", override)
    assert "outside the character directories" in str(excinfo.value)


def test_char_path_inside_a_registered_directory_is_allowed(pack):
    module, models = pack
    loader = module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"]()
    (char,) = loader.load("", str(models / "characters" / "Ada.char"))
    assert char.name == "Ada"


def test_applying_a_lora_the_model_cannot_receive_is_refused(pack):
    module, _ = pack
    from omnichar_sdk import CharError

    loader = module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"]()
    (char,) = loader.load("Ada.char")
    node = module.NODE_CLASS_MAPPINGS["OmnicharApplyCharacterLoRA"]()

    # A model with none of the adapter's modules.
    with pytest.raises(CharError) as excinfo:
        node.apply(FakeModel(["something.else"]), char, -1.0)
    message = str(excinfo.value)
    assert "0 of 1 modules" in message
    assert "0%" in message and "90%" in message
    assert "near miss" in message
    assert "lower min_key_coverage" in message


def test_applying_a_lora_the_model_accepts_uses_the_recorded_strength(pack):
    module, _ = pack
    loader = module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"]()
    (char,) = loader.load("Ada.char")
    node = module.NODE_CLASS_MAPPINGS["OmnicharApplyCharacterLoRA"]()
    model, _clip = node.apply(FakeModel(DOUBLE), char, -1.0)
    # -1 means "the strength the character recorded", which the fixture sets to 0.8.
    assert model[2] == pytest.approx(0.8)


def test_validate_inputs_applies_the_caps(pack, tmp_path):
    module, models = pack
    from conftest import MANIFEST, build

    cls = module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"]
    assert cls.VALIDATE_INPUTS("Ada.char") is True

    # This runs on every queue, before the node does, and was the one path that skipped limits.py.
    bomb = build(tmp_path, {"refs/000.png": b"\x00" * (8 * 1024 * 1024)}, MANIFEST)
    planted = models / "characters" / "bomb.char"
    planted.write_bytes(bomb.read_bytes())
    message = cls.VALIDATE_INPUTS("bomb.char")
    assert message is not True
    assert "expands" in message


def test_validate_inputs_rejects_a_path_outside_the_character_directories(pack):
    module, _ = pack
    cls = module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"]
    message = cls.VALIDATE_INPUTS("Ada.char", "/etc/passwd")
    assert message is not True
    assert "outside the character directories" in message


def test_decode_character_gives_conditioning_references_and_a_sheet(pack):
    module, _ = pack
    loader = module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"]()
    (char,) = loader.load("Ada.char")
    clip = FakeClip()

    cond, images, refs, sheet, prompt, _voice = module.NODE_CLASS_MAPPINGS[
        "OmnicharDecodeCharacter"
    ]().decode(char, "ordinal", clip, "in the rain")
    assert prompt.startswith("Images 1 and 2 show Ada,")
    assert prompt.endswith("in the rain")
    # The prompt reaches the encoder, which is the whole reason this node takes a CLIP.
    assert clip.seen == prompt
    assert cond[0][0].startswith("cond:Images 1 and 2")
    assert images.shape[0] == 2
    assert len(refs) == 2
    assert sheet.shape[0] == 1 and sheet.shape[3] == 3


def test_encode_character_round_trips_through_save_and_load(pack, tmp_path):
    import torch

    module, models = pack
    face = torch.rand(1, 96, 64, 3)
    body = torch.rand(2, 72, 128, 3)

    (char,) = module.NODE_CLASS_MAPPINGS["OmnicharEncodeCharacter"]().encode(
        "Bo", "A tall man with a shaved head.", 512, face=face, body=body
    )
    assert char.name == "Bo"
    assert [r.role for r in char.get_references()] == ["face", "body", "body"]
    # Compiled for both reference archs, so it applies without a rebuild.
    assert char.get_info().ref_archs == ["flux2-klein", "minimax-h3"]

    saved = module.NODE_CLASS_MAPPINGS["OmnicharSaveCharacter"]().save(char, "bo")
    # An OUTPUT_NODE returns a ui payload alongside its result, or the node looks like a no-op.
    assert saved["ui"]["text"] == list(saved["result"])
    written = Path(saved["result"][0])
    assert written.name == "bo.char" and written.parent == (models / "characters")

    from omnichar_sdk import Character

    assert Character.open(written).get_description() == "A tall man with a shaved head."


def test_saving_refuses_to_clobber_unless_told(pack):
    module, _ = pack
    from omnichar_sdk import CharError

    loader = module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"]()
    (char,) = loader.load("Ada.char")
    save = module.NODE_CLASS_MAPPINGS["OmnicharSaveCharacter"]()
    with pytest.raises(CharError) as excinfo:
        save.save(char, "Ada.char")
    assert "already exists" in str(excinfo.value)
    assert save.save(char, "Ada.char", overwrite=True)["result"][0].endswith("Ada.char")


def test_encode_with_no_images_says_what_to_wire(pack):
    module, _ = pack
    from omnichar_sdk import CharError

    with pytest.raises(CharError) as excinfo:
        module.NODE_CLASS_MAPPINGS["OmnicharEncodeCharacter"]().encode("X", "", 512)
    assert "at least one reference" in str(excinfo.value)


def test_the_shipped_workflows_match_the_nodes(pack):
    """A workflow that names a node we no longer register is a broken download."""
    import json

    for path in sorted((ROOT / "workflows").glob("*.json")):
        wf = json.loads(path.read_text())
        for node in wf["nodes"]:
            cls = module_for(pack, node["type"])
            if cls is None:
                continue
            spec = cls.INPUT_TYPES()
            widgets = [
                k
                for k, v in list(spec.get("required", {}).items())
                + list(spec.get("optional", {}).items())
                if not (isinstance(v[0], str) and v[0] in ("CHARACTER", "MODEL", "CLIP", "IMAGE"))
            ]
            values = node.get("widgets_values", [])
            assert len(values) <= len(widgets), f"{path.name}: {node['type']} {values} vs {widgets}"


def module_for(pack, node_type):
    module, _ = pack
    return module.NODE_CLASS_MAPPINGS.get(node_type)


def test_decode_works_with_no_clip_so_no_checkpoint_is_needed(pack):
    module, _ = pack
    loader = module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"]()
    (char,) = loader.load("Ada.char")

    cond, images, refs, sheet, prompt, _voice = module.NODE_CLASS_MAPPINGS[
        "OmnicharDecodeCharacter"
    ]().decode(char, "ordinal")
    # Decoding is extraction, so it must not require a model to be loaded first.
    assert cond is None
    assert images.shape[0] == 2 and len(refs) == 2 and sheet.shape[0] == 1
    assert prompt.startswith("Images 1 and 2 show Ada,")


def test_encode_takes_several_images_per_role(pack):
    """A ComfyUI input takes one link, so each role needs more than one slot."""
    import torch

    module, _ = pack
    (char,) = module.NODE_CLASS_MAPPINGS["OmnicharEncodeCharacter"]().encode(
        "Bo",
        "A tall man.",
        512,
        face=torch.rand(1, 96, 64, 3),
        face_2=torch.rand(1, 96, 64, 3),
        face_3=torch.rand(1, 96, 64, 3),
        body=torch.rand(1, 72, 128, 3),
        cloths_2=torch.rand(1, 64, 64, 3),
    )
    # Slots and batches both contribute, and roles stay grouped face then body then cloth.
    assert [r.role for r in char.get_references()] == ["face", "face", "face", "body", "cloth"]


def test_a_batch_in_one_slot_still_counts_as_several_references(pack):
    import torch

    module, _ = pack
    (char,) = module.NODE_CLASS_MAPPINGS["OmnicharEncodeCharacter"]().encode(
        "Bo", "A tall man.", 512, face=torch.rand(4, 96, 64, 3)
    )
    assert [r.role for r in char.get_references()] == ["face"] * 4


def test_saving_is_never_cached(pack):
    """A cached save is a save that never happens: ComfyUI reports 0.00s and writes nothing."""
    import math

    module, _ = pack
    cls = module.NODE_CLASS_MAPPINGS["OmnicharSaveCharacter"]
    assert math.isnan(cls.IS_CHANGED(None, "x.char"))


def test_saving_makes_the_new_character_visible_to_the_loader(pack):
    module, models = pack
    loader = module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"]()
    (char,) = loader.load("Ada.char")

    listed = lambda: set(  # noqa: E731
        module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"].INPUT_TYPES()["required"]["char"][0]
    )
    before = listed()
    module.NODE_CLASS_MAPPINGS["OmnicharSaveCharacter"]().save(char, "fresh")
    after = listed()
    # Without dropping ComfyUI's cached listing this stays stale until a restart.
    assert "fresh.char" in after - before
    assert (models / "characters" / "fresh.char").is_file()


def test_character_reference_returns_one_image_by_position(pack):
    """A model with numbered reference slots needs one image per slot, not a capped batch."""
    module, _ = pack
    loader = module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"]()
    (char,) = loader.load("Ada.char")
    refs = module.NODE_CLASS_MAPPINGS["OmnicharDecodeCharacter"]().decode(char, "token")[2]
    node = module.NODE_CLASS_MAPPINGS["OmnicharCharacterReference"]()

    first, role_a, count = node.pick(refs, 0)
    second, role_b, _ = node.pick(refs, 1)
    assert count == 2
    assert first.shape[0] == 1 and second.shape[0] == 1
    assert (role_a, role_b) == ("face", "body")
    # Different positions must be different images, which a capped batch would not guarantee.
    assert first.shape != second.shape


def test_a_position_past_the_end_says_what_the_last_one_is(pack):
    module, _ = pack
    from omnichar_sdk import CharError

    loader = module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"]()
    (char,) = loader.load("Ada.char")
    refs = module.NODE_CLASS_MAPPINGS["OmnicharDecodeCharacter"]().decode(char, "token")[2]
    with pytest.raises(CharError) as excinfo:
        module.NODE_CLASS_MAPPINGS["OmnicharCharacterReference"]().pick(refs, 4)
    message = str(excinfo.value)
    assert "The last one is 1" in message
    assert "Leave the extra slots" in message


def test_slots_and_prompt_cannot_disagree_about_the_reference_set(pack):
    """The point of routing through refs: one node decides the set, so numbering cannot drift."""
    module, _ = pack
    loader = module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"]()
    (char,) = loader.load("Ada.char")
    decode = module.NODE_CLASS_MAPPINGS["OmnicharDecodeCharacter"]()
    pick = module.NODE_CLASS_MAPPINGS["OmnicharCharacterReference"]()

    _, _, refs, _, prompt, _ = decode.decode(char, "token", arch="flux2-klein", max_references=1)
    # One reference compiled for flux2-klein, so the prompt names exactly one picture and the
    # only valid slot index is 0.
    assert prompt.startswith("<Picture 1> shows Ada,")
    assert "<Picture 2>" not in prompt
    assert pick.pick(refs, 0)[2] == 1
    with pytest.raises(Exception, match="index 1 does not exist"):
        pick.pick(refs, 1)


def test_reference_latents_are_attached_in_one_append(pack):
    """FLUX.2 takes references as latents on the conditioning, however many there are."""
    from comfystub import FakeVae

    module, _ = pack
    loader = module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"]()
    (char,) = loader.load("Ada.char")
    refs = module.NODE_CLASS_MAPPINGS["OmnicharDecodeCharacter"]().decode(char, "ordinal")[2]
    vae = FakeVae()

    base = [["cond", {}]]
    (out,) = module.NODE_CLASS_MAPPINGS["OmnicharCharacterReferenceLatent"]().attach(
        base, refs, vae
    )
    # Every reference reaches the VAE, at its own size, with alpha dropped.
    assert len(vae.encoded) == 2
    assert all(shape[0] == 1 and shape[3] == 3 for shape in vae.encoded)
    assert out[0][1]["reference_latents"] == ["latent1", "latent2"]
    # The original conditioning is not mutated, so it can feed another branch.
    assert base[0][1] == {}


def test_attaching_to_conditioning_that_already_has_references_appends(pack):
    from comfystub import FakeVae

    module, _ = pack
    loader = module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"]()
    (char,) = loader.load("Ada.char")
    refs = module.NODE_CLASS_MAPPINGS["OmnicharDecodeCharacter"]().decode(char, "ordinal")[2]

    (out,) = module.NODE_CLASS_MAPPINGS["OmnicharCharacterReferenceLatent"]().attach(
        [["cond", {"reference_latents": ["existing"]}]], refs, FakeVae()
    )
    assert out[0][1]["reference_latents"] == ["existing", "latent1", "latent2"]


def test_attaching_no_references_says_so(pack):
    from comfystub import FakeVae
    from omnichar_sdk import CharError

    module, _ = pack
    with pytest.raises(CharError, match="no references"):
        module.NODE_CLASS_MAPPINGS["OmnicharCharacterReferenceLatent"]().attach(
            [["cond", {}]], [], FakeVae()
        )


def test_references_split_puts_each_reference_on_its_own_output(pack):
    """One node per model, rather than one Character Reference node per slot."""
    module, _ = pack
    loader = module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"]()
    (char,) = loader.load("Ada.char")
    refs = module.NODE_CLASS_MAPPINGS["OmnicharDecodeCharacter"]().decode(char, "token")[2]

    out = module.NODE_CLASS_MAPPINGS["OmnicharCharacterReferencesSplit"]().split(refs)
    images, count = out[:-1], out[-1]
    assert count == 2
    assert images[0] is not None and images[1] is not None
    # Ada has two references, so the remaining slots are empty rather than padded with a blank.
    assert images[2] is images[3] is images[4] is None


def test_split_matches_picking_each_position_one_at_a_time(pack):
    import torch

    module, _ = pack
    loader = module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"]()
    (char,) = loader.load("Ada.char")
    refs = module.NODE_CLASS_MAPPINGS["OmnicharDecodeCharacter"]().decode(char, "token")[2]

    split = module.NODE_CLASS_MAPPINGS["OmnicharCharacterReferencesSplit"]().split(refs)
    pick = module.NODE_CLASS_MAPPINGS["OmnicharCharacterReference"]()
    for i in range(2):
        assert torch.equal(split[i], pick.pick(refs, i)[0])


# --- a character's voice --------------------------------------------------------------------------


def _speech(seconds, rate=24000, silence=0.0, channels=2):
    """A ComfyUI AUDIO: a tone after ``silence`` seconds of nothing, so the leading cut shows."""
    import math

    import torch

    total = int((silence + seconds) * rate)
    t = torch.arange(total) / rate
    tone = 0.3 * torch.sin(2 * math.pi * 220 * t)
    tone[: int(silence * rate)] = 0
    return {"waveform": tone.repeat(channels, 1).unsqueeze(0), "sample_rate": rate}


def _voiced(module, seconds=6.0, silence=0.0):
    import torch

    (char,) = module.NODE_CLASS_MAPPINGS["OmnicharEncodeCharacter"]().encode(
        "got", "", 512, face=torch.rand(1, 96, 64, 3), voice=_speech(seconds, silence=silence)
    )
    return char


def test_encode_stores_a_voice_where_omnichar_studio_reads_it(pack):
    module, _ = pack
    char = _voiced(module, seconds=6.0, silence=1.5)
    entry = char.manifest.reserved["voice"]
    assert entry["samples"][0]["path"] == "voice/samples/000.wav"
    assert entry["payload"]["path"] == "voice/payload/000.wav"
    # Studio rebuilds its own payload from the sample rather than trusting this one.
    assert entry["payload"]["encoder"] == {"id": "h3-voice-wav", "version": "sdk-1"}
    voice = char.get_voice()
    assert voice is not None and voice.channels == 1 and voice.sample_rate == 24000
    assert 6.0 <= voice.seconds < 6.2, "leading silence is cut, with a tenth of a second kept"
    assert char.get_info().voice_seconds == voice.seconds


def test_a_voice_too_short_to_carry_one_is_refused(pack):
    module, _ = pack
    from omnichar_sdk import CharError

    with pytest.raises(CharError, match="at least 3s"):
        _voiced(module, seconds=1.0)


def test_the_payload_is_capped_at_thirty_seconds(pack):
    module, _ = pack
    voice = _voiced(module, seconds=40.0).get_voice()
    assert voice is not None and voice.seconds == pytest.approx(30.0, abs=0.01)


def _decode(module, char, style="token", prompt='got says "never forget what you are"', **kw):
    return module.NODE_CLASS_MAPPINGS["OmnicharDecodeCharacter"]().decode(
        char, style, prompt=prompt, **kw
    )


def test_dialogue_sends_the_voice_and_names_it_in_the_prompt(pack):
    module, _ = pack
    *_, prompt, audio = _decode(module, _voiced(module))
    assert "<Audio 1> is got's voice. got speaks in this voice, lips moving in sync" in prompt
    assert audio["sample_rate"] == 24000 and audio["waveform"].shape[:2] == (1, 1)


def test_a_silent_scene_leaves_the_voice_out(pack):
    module, _ = pack
    *_, prompt, audio = _decode(module, _voiced(module), prompt="got walks along the lake")
    assert audio is None and "<Audio" not in prompt


def test_always_and_never_override_the_prompt(pack):
    module, _ = pack
    char = _voiced(module)
    assert _decode(module, char, prompt="got walks", voice="always")[-1] is not None
    assert _decode(module, char, voice="never")[-1] is None


def test_the_voice_position_follows_the_setting(pack):
    module, _ = pack
    *_, prompt, _audio = _decode(module, _voiced(module), audio_position=2)
    assert "<Audio 2> is got's voice." in prompt


def test_only_the_h3_style_takes_a_voice(pack):
    """FLUX.2 has no audio input, so naming <Audio 1> would point at nothing."""
    module, _ = pack
    *_, prompt, audio = _decode(module, _voiced(module), style="ordinal")
    assert audio is None and "<Audio" not in prompt


def test_a_character_without_a_voice_decodes_exactly_as_before(pack):
    module, _ = pack
    loader = module.NODE_CLASS_MAPPINGS["OmnicharLoadCharacter"]()
    (char,) = loader.load("Ada.char")
    assert char.get_voice() is None and char.get_info().voice_seconds is None
    *_, prompt, audio = _decode(module, char)
    assert audio is None
    assert prompt == f'{char.get_prompt(style="token")}got says "never forget what you are"'.strip()


def test_a_saved_voice_survives_a_reopen(pack):
    module, _ = pack
    saved = module.NODE_CLASS_MAPPINGS["OmnicharSaveCharacter"]().save(_voiced(module), "got")
    from omnichar_sdk import Character

    reopened = Character.open(saved["result"][0])
    voice = reopened.get_voice()
    assert voice is not None and voice.read_wav()[:4] == b"RIFF"
