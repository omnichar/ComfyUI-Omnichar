# SPDX-License-Identifier: GPL-3.0-or-later
"""Apply a character's adapter, refusing one the model cannot receive; a half match near-misses."""

from __future__ import annotations

import logging

import comfy.lora
import comfy.sd
from omnichar_sdk import KEY_WRAPPERS, CharChanged, CharError, Portability, module_stem

from .common import CATEGORY, CHARACTER_INPUT, fail_on_change, require_usable

logger = logging.getLogger("omnichar")

def _module_stems(keys) -> set[str]:
    """The modules a set of adapter tensor names attaches to, wrappers stripped."""
    return {stem for key in keys if (stem := module_stem(key)) is not None}


def _target_stems(model, clip) -> set[str]:
    """Every module the model accepts an adapter on, with kohya's underscored form added too."""
    key_map: dict[str, object] = {}
    if model is not None:
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
    stems: set[str] = set()
    for key in key_map:
        stripped = key
        for wrapper in KEY_WRAPPERS:
            if stripped.startswith(wrapper):
                stripped = stripped[len(wrapper) :]
                break
        stems.add(stripped)
        if "." not in stripped:
            stems.add(stripped.replace("_", "."))
    return stems


def _coverage(adapter_keys, model, clip) -> tuple[float, int, int]:
    ours = _module_stems(adapter_keys)
    if not ours:
        return (0.0, 0, 0)
    theirs = _target_stems(model, clip)
    matched = sum(1 for stem in ours if stem in theirs or any(t.endswith(stem) for t in theirs))
    return (matched / len(ours), matched, len(ours))


class OmnicharApplyCharacterLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "char": CHARACTER_INPUT,
                "strength": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 4.0,
                        "step": 0.01,
                        "tooltip": (
                            "How strongly the adapter applies. -1 uses the strength recorded in "
                            "the character, which is what it was judged at."
                        ),
                    },
                ),
            },
            "optional": {
                "clip": ("CLIP",),
                "arch": (
                    "STRING",
                    {"default": "", "tooltip": "Which adapter, if the character has several."},
                ),
                "min_key_coverage": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "advanced": True,
                        "tooltip": (
                            "Refuse the adapter if less than this share of its modules match the "
                            "model. Lowering it applies a partial adapter on purpose."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Apply a character's trained LoRA to a model. Refuses rather than applying an adapter the "
        "model cannot receive, because a partial match renders a near miss instead of failing."
    )

    def apply(self, model, char, strength, clip=None, arch="", min_key_coverage=0.9):
        require_usable(char)
        try:
            lora = char.get_lora(arch or None)
        except CharChanged as error:
            raise fail_on_change(error) from error
        if lora is None:
            raise CharError(
                f"{char.name} carries no trained adapter"
                + (f" for {arch}" if arch else "")
                + ". Use the reference nodes instead, or train one in Omnichar Studio."
            )

        if lora.portability is Portability.INCOMPATIBLE:
            raise CharError(
                f"{char.name}'s {lora.arch} adapter cannot be applied here. {lora.reason} "
                "Lowering min_key_coverage will not help, because no keys match at all."
            )

        # From the header alone, so an adapter that will be refused is never pulled into memory.
        coverage, matched, total = _coverage(lora.keys(), model, clip)

        if coverage < min_key_coverage:
            skipped = 100.0 * (1.0 - coverage)
            raise CharError(
                f"{char.name}'s {lora.arch} adapter matches {matched} of {total} modules on "
                f"this model, which is {coverage:.0%}. The minimum is {min_key_coverage:.0%}.\n"
                f"Applying it would skip {skipped:.0f}% of the adapter and render a near miss "
                f"rather than failing, which is why it stops here.\n"
                f"{lora.reason}\n"
                "Either load the model this adapter was trained on "
                f"({lora.base or 'not recorded'}), or lower min_key_coverage to accept a partial "
                "result on purpose."
            )

        from safetensors.torch import load as load_safetensors

        # safetensors' byte loader: no pickle, unlike torch.load, and no temporary file.
        state = load_safetensors(lora.bytes())
        applied = lora.strength if strength < 0 else strength
        if coverage < 1.0:
            logger.warning(
                "Omnichar: %s's %s adapter matches %d of %d modules (%.0f%%); %.0f%% is skipped.",
                char.name, lora.arch, matched, total, coverage * 100, (1 - coverage) * 100,
            )
        patched_model, patched_clip = comfy.sd.load_lora_for_models(
            model, clip, state, applied, applied if clip is not None else 0.0
        )
        return (patched_model, patched_clip if clip is not None else clip)
