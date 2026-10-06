# SPDX-License-Identifier: GPL-3.0-or-later
"""Build a character from reference images, and write one to disk."""

from __future__ import annotations

import logging
from pathlib import Path

from omnichar_sdk import (
    FLUX2_KLEIN_ARCH,
    MINIMAX_H3_ARCH,
    VOICE_MAX_SECONDS,
    Character,
    CharError,
    encode_character,
    safe_output_name,
    set_voice,
    write,
)

from . import folders
from .common import CATEGORY, CHARACTER_INPUT, audio_to_voice, from_image

logger = logging.getLogger("omnichar")

#: Several slots per role, because a ComfyUI input takes one link. Each still accepts a batch, so
#: three Load Image nodes or one batch of three both give three references.
SLOTS_PER_ROLE = 3

_ROLE_HINTS = {
    "face": "Face references. Identity comes from these.",
    "body": "Full-body references, for build and proportions.",
    "cloths": "Wardrobe references.",
}
_ROLE_SLOTS = [
    (slot if n == 1 else f"{slot}_{n}", "cloth" if slot == "cloths" else slot)
    for slot in _ROLE_HINTS
    for n in range(1, SLOTS_PER_ROLE + 1)
]


def _role_inputs() -> dict[str, object]:
    out: dict[str, object] = {}
    for name, _role in _ROLE_SLOTS:
        base = name.split("_")[0]
        out[name] = ("IMAGE", {"tooltip": _ROLE_HINTS[base]})
    return out


class OmnicharEncodeCharacter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "Character"}),
                "description": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "What the character looks like. Every model reads this, and a LoRA "
                            "binds to it, so write it once and keep it."
                        ),
                    },
                ),
                "resolution": (
                    "INT",
                    {
                        "default": 512,
                        "min": 0,
                        "max": 4096,
                        "step": 64,
                        "tooltip": (
                            "What the compiled reference sets are stored at. 0 uses each model's "
                            "own budget. Originals are always kept untouched."
                        ),
                    },
                ),
            },
            "optional": {
                **_role_inputs(),
                "voice": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Optional. About 30 seconds of the character speaking, no music. "
                            "Only MiniMax H3 Reference to Video uses it."
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("CHARACTER",)
    RETURN_NAMES = ("char",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Build a character from reference images, and optionally a voice. Compiles a reference set "
        "for FLUX.2 and MiniMax H3, so it applies on any model that takes references."
    )

    def encode(self, name, description, resolution, voice=None, **images):
        pairs = []
        for slot, role in _ROLE_SLOTS:
            batch = images.get(slot)
            if batch is None:
                continue
            pairs.extend((image, role) for image in from_image(batch))
        if not pairs:
            raise CharError(
                "A character needs at least one reference image. Wire a face, body or cloths "
                "input on this node."
            )
        doc = encode_character(
            name,
            description,
            pairs,
            archs=(FLUX2_KLEIN_ARCH, MINIMAX_H3_ARCH),
            resolution=resolution or None,
        )
        if voice is not None:
            sample, payload = audio_to_voice(voice, VOICE_MAX_SECONDS)
            set_voice(doc, sample, payload)
        return (Character.from_bytes(_to_bytes(doc), f"{name}.char"),)


def _to_bytes(doc) -> bytes:
    """Through a temporary file, so the character in memory is the bytes that would be saved."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        return write(Path(tmp) / "c.char", doc).read_bytes()


class OmnicharSaveCharacter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "char": CHARACTER_INPUT,
                "filename": (
                    "STRING",
                    {
                        "default": "model.char",
                        "tooltip": "Written into the characters folder. .char is added if missing.",
                    },
                ),
            },
            "optional": {
                "overwrite": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Off by default, so a character is not lost."},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "Write a character into the characters folder so the loader can pick it up."

    @classmethod
    def IS_CHANGED(cls, char, filename, overwrite=False):
        # Always. Writing a file is a side effect, and a cached save is a save that never happens:
        # ComfyUI reports "executed in 0.00 seconds" and nothing reaches disk.
        return float("nan")

    def save(self, char, filename, overwrite=False):
        roots = folders.roots()
        if not roots:
            raise CharError(
                "No characters directory is registered. Create ComfyUI/models/characters and "
                "restart."
            )
        target = roots[0] / safe_output_name(filename, ".char")
        if target.exists() and not overwrite:
            raise CharError(
                f"{target.name} already exists. Turn on overwrite, or choose another filename."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(char.to_bytes())
        folders.forget_listing()
        logger.info("Omnichar: wrote %s (%d bytes)", target, target.stat().st_size)
        # Without a ui payload an OUTPUT_NODE shows nothing, which reads as the node doing nothing.
        return {"ui": {"text": [str(target)]}, "result": (str(target),)}
