# SPDX-License-Identifier: Apache-2.0
"""Read Omnichar Studio ``.char`` character files; images need the ``images`` extra."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version

from .character import Character, CharacterInfo
from .charfile import (
    FORMAT_VERSION,
    MAGIC,
    ORIGIN_HARVESTED,
    ORIGIN_ORIGINAL,
    ROLE_BODY,
    ROLE_CLOTH,
    ROLE_FACE,
    ROLES,
    CharChanged,
    CharDoc,
    CharError,
    Manifest,
    looks_like_char,
    payload_key,
    read,
    write,
)
from .encode import (
    FAL_REF_ARCH,
    FLUX2_KLEIN_ARCH,
    MINIMAX_H3_ARCH,
    REFERENCE_POLICIES,
    build_payload,
    encode_character,
    normalise_reference,
)
from .lora import (
    KEY_SUFFIXES,
    KEY_WRAPPERS,
    Lora,
    Portability,
    module_stem,
    safe_output_name,
)
from .prompt import STYLES, prompt_prefix
from .references import FIT_MODES, SIZE_POLICIES, Reference, allocate_roles, common_size, fit_roles
from .sheet import reference_sheet, sheet_png
from .voice import MAX_SECONDS as VOICE_MAX_SECONDS
from .voice import VOICE_MODES, Voice, drop_voice, set_voice, speaks, wanted

try:
    __version__ = _installed_version("omnichar-sdk")
except PackageNotFoundError:  # running from a source tree with nothing installed
    __version__ = "0.0.0+source"

__all__ = [
    "write",
    "VOICE_MAX_SECONDS",
    "VOICE_MODES",
    "Voice",
    "drop_voice",
    "set_voice",
    "speaks",
    "wanted",
    "normalise_reference",
    "encode_character",
    "build_payload",
    "REFERENCE_POLICIES",
    "MINIMAX_H3_ARCH",
    "FLUX2_KLEIN_ARCH",
    "FAL_REF_ARCH",
    "FIT_MODES",
    "KEY_SUFFIXES",
    "KEY_WRAPPERS",
    "FORMAT_VERSION",
    "MAGIC",
    "ORIGIN_HARVESTED",
    "ORIGIN_ORIGINAL",
    "ROLES",
    "ROLE_BODY",
    "ROLE_CLOTH",
    "ROLE_FACE",
    "SIZE_POLICIES",
    "STYLES",
    "CharChanged",
    "CharDoc",
    "CharError",
    "Character",
    "CharacterInfo",
    "Lora",
    "Manifest",
    "Portability",
    "Reference",
    "__version__",
    "allocate_roles",
    "common_size",
    "fit_roles",
    "looks_like_char",
    "module_stem",
    "payload_key",
    "prompt_prefix",
    "reference_sheet",
    "sheet_png",
    "read",
    "safe_output_name",
]
