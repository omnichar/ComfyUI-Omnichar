# SPDX-License-Identifier: GPL-3.0-or-later
"""Unpack a character into the references, sheet and prompt the rest of a graph can use."""

from __future__ import annotations

from omnichar_sdk import (
    FIT_MODES,
    SIZE_POLICIES,
    STYLES,
    VOICE_MODES,
    CharChanged,
    CharError,
    common_size,
    reference_sheet,
    wanted,
)

from .common import (
    CATEGORY,
    CHARACTER_INPUT,
    REFS,
    fail_on_change,
    to_image,
    wav_to_audio,
)

_ARCH_TOOLTIP = (
    "Which reference set to send. 'originals' is what the character was built from; naming a "
    "model family sends the set compiled for it."
)


class OmnicharDecodeCharacter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "char": CHARACTER_INPUT,
                "style": (
                    list(STYLES),
                    {
                        "default": "ordinal",
                        "tooltip": (
                            "How the model addresses reference positions. FLUX.2 reads ordinal "
                            "prose, MiniMax H3 reads <Picture N>, Seedance reads @ImageN. "
                            "description-only drops positions, which is what a LoRA needs."
                        ),
                    },
                ),
            },
            "optional": {
                "clip": (
                    "CLIP",
                    {
                        "tooltip": (
                            "Optional. Wire one to get CONDITIONING straight out; leave it empty "
                            "and use the prompt output instead."
                        )
                    },
                ),
                "prompt": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "Appended after the character."},
                ),
                "arch": ("STRING", {"default": "originals", "tooltip": _ARCH_TOOLTIP}),
                "max_references": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 64,
                        "tooltip": (
                            "Cap the number sent. Slots divide between face, body and outfit "
                            "rather than cutting the end, so a character keeps its wardrobe. "
                            "0 means no cap."
                        ),
                    },
                ),
                "size_from": (list(SIZE_POLICIES), {"default": "first"}),
                "fit": (list(FIT_MODES), {"default": "pad"}),
                "voice": (
                    list(VOICE_MODES),
                    {
                        "default": "auto",
                        "tooltip": (
                            "When to send the character's voice. auto sends it when this node's "
                            "prompt has dialogue, a quoted line or a word like says or speaks. "
                            "Only the token style (MiniMax H3) uses a voice."
                        ),
                    },
                ),
                "audio_position": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 3,
                        "tooltip": (
                            "Which <Audio N> the voice is. 1 when it goes into ref_audio_0 and "
                            "no reference video carries sound."
                        ),
                    },
                ),
            },
        }

    # `voice` is last, so a workflow saved before it existed keeps every link it had.
    RETURN_TYPES = ("CONDITIONING", "IMAGE", REFS, "IMAGE", "STRING", "AUDIO")
    RETURN_NAMES = ("conditioning", "references", "refs", "sheet", "prompt", "voice")
    FUNCTION = "decode"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Unpack a character into reference images, a numbered contact sheet, its prompt and its "
        "voice. Wire only what you need. A CLIP is optional and only needed for the conditioning "
        "output."
    )

    def decode(
        self, char, style, clip=None, prompt="", arch="originals",
        max_references=0, size_from="first", fit="pad", voice="auto", audio_position=1,
    ):
        try:
            selected = None if arch in ("", "originals") else arch
            refs = char.get_references(
                arch=selected, limit=None if max_references <= 0 else max_references
            )
            if not refs:
                raise CharError(
                    f"{char.name} has no references"
                    + (f" compiled for {arch}" if selected else "")
                    + ". Change the reference set on this node."
                )
            # Only H3 reads a voice, and it addresses positions in the token style.
            stored = char.get_voice() if style == "token" else None
            send = stored is not None and wanted(voice, prompt)
            text = char.get_prompt(
                style=style,
                arch=selected,
                count=len(refs),
                voice_position=audio_position if send else None,
            )
            full = f"{text}{prompt}".strip()
            size = common_size(refs, size_from)
            images = to_image([ref.fit(size, fit) for ref in refs])
            # Built from the same resolved list, so the numbers on the sheet are the numbers the
            # prompt uses and the positions the slots receive.
            sheet = to_image([reference_sheet(refs, title=char.name)])
            audio = wav_to_audio(stored.read_wav()) if send and stored is not None else None
        except CharChanged as error:
            raise fail_on_change(error) from error
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full)) if clip else None
        return (conditioning, images, refs, sheet, full, audio)
