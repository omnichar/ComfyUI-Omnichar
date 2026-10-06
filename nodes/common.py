# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared pieces for every node: the socket type, image conversion, and the mid-run failure."""

from __future__ import annotations

from typing import Any

import torch
from omnichar_sdk import Character, CharChanged, CharError, Reference

#: forceInput on every CHARACTER input, or the frontend draws a widget for a type it cannot know.
CHARACTER = "CHARACTER"
CHARACTER_INPUT = (CHARACTER, {"forceInput": True})

#: The resolved reference list, already filtered and ordered. Carried as its own type so one node
#: decides which references a model gets; a second node reading the character again could pick a
#: different set and the prompt's numbers would stop matching the slots.
REFS = "CHARACTER_REFS"
REFS_INPUT = (REFS, {"forceInput": True})

CATEGORY = "Omnichar"


def to_image(images: list[Any]) -> torch.Tensor:
    """Pillow images to a ComfyUI IMAGE batch: float 0 to 1, shape [B, H, W, 3]."""
    import numpy as np

    if not images:
        raise CharError("No references to convert. Check the role filter and the maximum.")
    frames = [torch.from_numpy(np.array(image, dtype=np.float32) / 255.0) for image in images]
    return torch.stack(frames)


def from_image(batch: Any) -> list[Any]:
    """A ComfyUI IMAGE batch back to Pillow images, one per frame."""
    import numpy as np
    from PIL import Image

    out = []
    for frame in batch:
        array = (frame.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        out.append(Image.fromarray(array, "RGB"))
    return out


def fail_on_change(error: CharChanged) -> CharError:
    """Fail a render whose character was replaced mid-run; retrying would use a different one."""
    return CharError(
        "The character file changed while this render was running. "
        f"Queue it again to use the new version. ({error})"
    )


def references_of(
    char: Character, arch: str, role: str, maximum: int
) -> list[Reference]:
    """References for a node's settings, with the widget sentinels turned into arguments."""
    refs = char.get_references(
        arch=None if arch in ("", "originals") else arch,
        role=None if role == "any" else role,
        limit=None if maximum <= 0 else maximum,
    )
    if not refs:
        raise CharError(
            f"{char.name} has no {role} references"
            + (f" compiled for {arch}" if arch not in ("", "originals") else "")
            + ". Change the role or the reference set on this node."
        )
    return refs


#: Below this is silence for the leading cut: -50 dB, as Omnichar Core's ffmpeg filter uses.
_SILENCE = 10 ** (-50 / 20)
#: Kept before the first sound, so a voice does not open on a clipped consonant.
_LEAD_SECONDS = 0.1


def _pcm16(waveform: torch.Tensor, rate: int) -> bytes:
    """A [channels, samples] float tensor as PCM16 WAV bytes."""
    import io
    import wave

    import numpy as np

    pcm = (waveform.clamp(-1.0, 1.0).cpu().numpy().T * 32767.0).astype(np.int16)
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setnchannels(int(waveform.shape[0]))
        handle.setsampwidth(2)
        handle.setframerate(int(rate))
        handle.writeframes(np.ascontiguousarray(pcm).tobytes())
    return out.getvalue()


def audio_to_voice(audio: dict[str, Any], max_seconds: float) -> tuple[bytes, bytes]:
    """A ComfyUI AUDIO as a voice: the sample as given, and the mono payload a model hears."""
    waveform = audio.get("waveform") if isinstance(audio, dict) else None
    rate = int(audio.get("sample_rate") or 0) if isinstance(audio, dict) else 0
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3 or rate <= 0:
        raise CharError("The voice input is not audio. Wire a Load Audio node into it.")
    clip = waveform[0].float()
    sample = _pcm16(clip, rate)
    mono = clip.mean(dim=0, keepdim=True)
    # H3 cuts an audio reference to the clip's length, so a 5s clip hears only the first 5s.
    loud = torch.nonzero(mono[0].abs() > _SILENCE)
    start = max(0, int(loud[0].item()) - int(_LEAD_SECONDS * rate)) if loud.numel() else 0
    payload = mono[:, start : start + int(max_seconds * rate)]
    return sample, _pcm16(payload, rate)


def wav_to_audio(data: bytes) -> dict[str, Any]:
    """PCM16 WAV bytes as a ComfyUI AUDIO: {"waveform": [1, channels, samples], "sample_rate"}."""
    import io
    import wave

    import numpy as np
    from omnichar_sdk.voice import wav_info

    # The header first: width, channels and rate are checked before any frame is decoded.
    _seconds, rate, channels = wav_info(data, "The character's voice")
    with wave.open(io.BytesIO(data), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    usable = len(frames) - len(frames) % (2 * channels)
    if usable <= 0:
        raise CharError(
            "The character's voice has no sound in it. Rebuild the character with a voice clip."
        )
    pcm = np.frombuffer(frames[:usable], dtype=np.int16).reshape(-1, channels).T.astype(np.float32)
    return {"waveform": torch.from_numpy(pcm / 32768.0).unsqueeze(0), "sample_rate": rate}
