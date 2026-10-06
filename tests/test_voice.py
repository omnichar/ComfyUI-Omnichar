"""The voice half of the format and the rule for sending it, mirrored from Omnichar Core."""

import pytest
from omnichar_sdk import prompt_prefix, speaks, wanted


@pytest.mark.parametrize(
    "prompt",
    [
        'got standing on a cliff says "Never forget what you are"',
        "Ada speaks to the camera",
        "she whispers to him",
        "Ada: “Hello there”",
    ],
)
def test_dialogue_switches_the_voice_on(prompt):
    assert speaks(prompt)


@pytest.mark.parametrize("prompt", ["Ada walks along the lake", "the knight's armour gleams"])
def test_a_silent_scene_leaves_it_off(prompt):
    assert not speaks(prompt)


def test_the_setting_overrides_the_prompt():
    assert wanted("always", "Ada walks") and not wanted("never", "Ada says hi")
    with pytest.raises(ValueError, match="Unknown voice setting"):
        wanted("sometimes", "")


def test_the_voice_line_is_omnichar_cores_word_for_word():
    """characters/apply.py writes this exact sentence; a different one is a different prompt."""
    text = prompt_prefix("got", "brown hair", ["face"], style="token", voice_position=1)
    assert text == (
        "<Picture 1> shows got, the same character in every image. <Audio 1> is got's voice. "
        "got speaks in this voice, lips moving in sync with every word. brown hair. "
    )


def test_no_voice_position_leaves_the_prompt_unchanged():
    assert "<Audio" not in prompt_prefix("got", "", ["face"], style="token")


# --- reading a voice from a file nobody vouches for --------------------------------------------

import copy
import io
import os
import wave

from conftest import MANIFEST, build
from omnichar_sdk import Character, CharError
from omnichar_sdk import voice as voice_module


def _wav(seconds=4.0, rate=16000, channels=1, width=2):
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        # Noise, not a tone: a constant clip compresses past the zip-bomb guard before any voice check.
        handle.writeframes(os.urandom(int(seconds * rate) * channels * width))
    return out.getvalue()


def _manifest(payload_path="voice/payload/000.wav", sample_path="voice/samples/000.wav"):
    manifest = copy.deepcopy(MANIFEST)
    manifest["reserved"] = {
        "voice": {
            "samples": [{"path": sample_path, "sha256": "x"}],
            "payload": {"path": payload_path, "seconds": 4.0},
        }
    }
    return manifest


def test_a_voice_pointing_outside_its_folder_is_refused(tmp_path):
    """Without the check a manifest could hand the adapter, or anything, to the audio decoder."""
    path = build(tmp_path, {"loras/a.safetensors": b"x"}, _manifest("loras/a.safetensors"))
    with pytest.raises(CharError, match="outside its voice folder"):
        Character.open(path).get_voice()


def test_an_oversized_voice_member_is_refused_before_it_is_read(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_module, "MAX_PAYLOAD_BYTES", 100)
    path = build(tmp_path, {"voice/payload/000.wav": _wav(), "voice/samples/000.wav": _wav()}, _manifest())
    voice = Character.open(path).get_voice()
    assert voice is not None
    with pytest.raises(CharError, match="over the 100"):
        voice.read_wav()


def test_a_payload_that_is_not_16_bit_pcm_is_named_not_crashed_on():
    with pytest.raises(CharError, match="16-bit mono or stereo"):
        voice_module.wav_info(_wav(width=1))
    with pytest.raises(CharError, match="not a WAV file"):
        voice_module.wav_info(b"not a wav at all")


def test_set_voice_refuses_what_omnichar_core_would_not_store():
    from omnichar_sdk import CharDoc, Manifest

    doc = CharDoc(manifest=Manifest(char_id="x", name="Ada", created_at=0, modified_at=0))
    with pytest.raises(CharError, match="mono"):
        voice_module.set_voice(doc, _wav(), _wav(channels=2))
    with pytest.raises(CharError, match="at most 30s"):
        voice_module.set_voice(doc, _wav(), _wav(seconds=40))
    assert voice_module.voice_entry(doc.manifest) is None
