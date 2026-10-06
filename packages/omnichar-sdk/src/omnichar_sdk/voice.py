# SPDX-License-Identifier: Apache-2.0
"""A character's voice, laid out as Omnichar Studio writes it, and the rule for when to send it."""

from __future__ import annotations

import io
import re
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from . import charfile as cf

#: In `reserved`, because an older reader rewriting the manifest drops unknown top-level keys.
VOICE_KEY = "voice"
VOICE_VERSION = 1
ENCODER_ID = "h3-voice-wav"
#: Not Studio's "1", so Studio rebuilds the payload with ffmpeg from the sample on first apply.
SDK_ENCODER_VERSION = "sdk-1"

#: Omnichar Core refuses less sound than this, and an H3 clip never hears more than this.
MIN_SECONDS = 3.0
MAX_SECONDS = 30.0
#: Omnichar Core's own cap on a voice upload.
MAX_SAMPLE_BYTES = 50 * 1024 * 1024
#: 30s of 16-bit stereo at 192 kHz, the most a payload this format writes could need.
MAX_PAYLOAD_BYTES = 24 * 1024 * 1024
#: Mono or stereo at a real audio rate; anything else is not a voice clip.
_CHANNELS = (1, 2)
_RATES = (8000, 192000)

#: Values of a node's voice setting.
VOICE_MODES = ("auto", "always", "never")

_SAMPLES = "voice/samples"
_PAYLOAD = "voice/payload"

#: Dialogue in a prompt: a double-quoted line, or a verb of speech. Mirrors Omnichar Core.
_QUOTED = re.compile(r'["“][^"”]{2,}["”]')
_SPEECH_VERBS = re.compile(
    r"\b(say|says|said|saying|speak|speaks|spoke|speaking|talk|talks|talking|tell|tells|told|"
    r"ask|asks|asked|reply|replies|replied|answer|answers|whisper|whispers|whispered|shout|shouts|"
    r"shouted|yell|yells|yelled|exclaim|exclaims|announce|announces|declare|declares|mutter|"
    r"mutters|sing|sings|singing|dialogue|monologue)\b",
    re.IGNORECASE,
)


def speaks(prompt: str) -> bool:
    """Whether a prompt has a character talking, which is when a voice belongs in the render."""
    return bool(_QUOTED.search(prompt) or _SPEECH_VERBS.search(prompt))


def wanted(mode: str, prompt: str) -> bool:
    """Whether to send the voice for this prompt: ``auto`` sends it only with dialogue."""
    if mode not in VOICE_MODES:
        raise ValueError(f"Unknown voice setting {mode!r}. Use one of {', '.join(VOICE_MODES)}.")
    return mode == "always" or (mode == "auto" and speaks(prompt))


def voice_line(name: str, position: int) -> str:
    """The sentence binding the voice to the character, word for word as Omnichar Core writes it."""
    return (
        f" <Audio {position}> is {name}'s voice. {name} speaks in this voice,"
        " lips moving in sync with every word."
    )


@dataclass(frozen=True)
class Voice:
    """A character's stored voice. The payload is the clip a model is given."""

    seconds: float
    sample_rate: int
    channels: int
    source_name: str
    payload_path: str
    sample_path: str
    _read: Callable[[str], bytes] = field(repr=False, compare=False)

    def read_wav(self) -> bytes:
        """The payload WAV, mono PCM with leading silence already cut."""
        return self._read(self.payload_path)

    def read_sample(self) -> bytes:
        """The clip the voice was built from, exactly as it was stored."""
        return self._read(self.sample_path)


def _dict(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def voice_entry(manifest: cf.Manifest) -> dict[str, Any] | None:
    """The manifest's voice record, or None for a character without one."""
    entry = _dict(manifest.reserved.get(VOICE_KEY))
    samples = entry.get("samples")
    if not isinstance(samples, list) or not samples:
        return None
    return entry


def voice_of(manifest: cf.Manifest, read: Callable[[str], bytes]) -> Voice | None:
    """The stored voice, or None; refuses a record pointing outside ``voice/``."""
    entry = voice_entry(manifest)
    if entry is None:
        return None
    payload = _dict(entry.get("payload"))
    sample = _dict(cast(list[Any], entry["samples"])[0])
    path = str(payload.get("path") or "")
    if not path:
        return None
    # A manifest is untrusted, and without this one could hand a LoRA to the audio decoder.
    for member in (path, str(sample.get("path") or "")):
        if not member.startswith("voice/") or ".." in member:
            raise cf.CharError(
                f"This character's voice points at {member!r}, outside its voice folder. Do not "
                "use it; get the character from a source you trust."
            )
    return Voice(
        seconds=float(payload.get("seconds") or 0),
        sample_rate=int(payload.get("sample_rate") or 0),
        channels=int(payload.get("channels") or 1),
        source_name=str(sample.get("source_name") or ""),
        payload_path=path,
        sample_path=str(sample.get("path") or ""),
        _read=read,
    )


def wav_info(data: bytes, label: str = "The voice clip") -> tuple[float, int, int]:
    """Seconds, sample rate and channels of a 16-bit PCM WAV, read from its header alone."""
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            rate, channels = handle.getframerate(), handle.getnchannels()
            width, frames = handle.getsampwidth(), handle.getnframes()
    except (wave.Error, EOFError) as error:
        raise cf.CharError(
            f"{label} is not a WAV file. Re-export the voice as 16-bit PCM WAV."
        ) from error
    if width != 2 or channels not in _CHANNELS or not _RATES[0] <= rate <= _RATES[1]:
        raise cf.CharError(
            f"{label} is {width * 8}-bit, {channels} channel(s) at {rate} Hz. Use 16-bit mono or "
            "stereo WAV at a normal sample rate."
        )
    return (frames / rate, rate, channels)


def set_voice(
    doc: cf.CharDoc, sample: bytes, payload: bytes, source_name: str = "voice.wav"
) -> None:
    """Store a voice on ``doc``, replacing any it had: both 16-bit WAV, the payload mono."""
    if len(sample) > MAX_SAMPLE_BYTES:
        raise cf.CharError(
            f"That voice is {len(sample) // 1024**2} MB; the limit is "
            f"{MAX_SAMPLE_BYTES // 1024**2} MB. Trim it to about 30 seconds of speech."
        )
    wav_info(sample, "The voice sample")
    seconds, rate, channels = wav_info(payload, "The voice payload")
    if channels != 1 or seconds > MAX_SECONDS + 0.5:
        raise cf.CharError(
            f"The voice payload must be mono and at most {MAX_SECONDS:.0f}s. Downmix and trim it "
            "before storing it."
        )
    if seconds < MIN_SECONDS:
        raise cf.CharError(
            f"That voice has {seconds:.1f}s of sound; it needs at least {MIN_SECONDS:.0f}s, and "
            "about 30s works best."
        )
    drop_voice(doc)
    sample_member = cf.member_name(_SAMPLES, 0, ".wav")
    payload_member = cf.member_name(_PAYLOAD, 0, ".wav")
    doc.members[sample_member] = sample
    doc.members[payload_member] = payload
    sample_sha = cf.sha256_bytes(sample)
    doc.manifest.reserved[VOICE_KEY] = {
        "version": VOICE_VERSION,
        "samples": [
            {
                "path": sample_member,
                "sha256": sample_sha,
                # A user's filename is data, never a path anything follows.
                "source_name": Path(source_name).name[:200],
                "bytes": len(sample),
            }
        ],
        "payload": {
            "payload_version": 1,
            "type": "voice",
            "encoder": {"id": ENCODER_ID, "version": SDK_ENCODER_VERSION},
            "source_sha256": sample_sha,
            "path": payload_member,
            "sha256": cf.sha256_bytes(payload),
            "sample_rate": rate,
            "channels": channels,
            "seconds": round(seconds, 3),
        },
        "scoring": {},
    }
    doc.manifest.modified_at = int(time.time())


def drop_voice(doc: cf.CharDoc) -> bool:
    """Remove the voice and its members. True when there was one."""
    entry = doc.manifest.reserved.pop(VOICE_KEY, None)
    for name in [n for n in doc.members if n.startswith("voice/")]:
        doc.members.pop(name, None)
    return entry is not None
