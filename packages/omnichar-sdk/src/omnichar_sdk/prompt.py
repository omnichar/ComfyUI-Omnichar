# SPDX-License-Identifier: Apache-2.0
"""Text binding a character to its reference positions, pinned to Omnichar's by golden strings."""

from __future__ import annotations

from collections.abc import Sequence

from .charfile import ROLE_BODY, ROLE_CLOTH, ROLE_FACE, ROLES
from .voice import voice_line

#: The addressing form each model family was trained on.
STYLES = ("ordinal", "token", "at-image", "description-only")

#: Naming only: "slim build" would be text competing with the reference images.
_ROLE_BINDINGS: dict[str, str] = {
    ROLE_FACE: "face",
    ROLE_BODY: "full body and build",
    ROLE_CLOTH: "outfit",
}


def _positions(style: str, numbers: list[int]) -> str:
    """How a role line refers *back* to positions, in prose, never re-declaring them."""
    # Never `<Picture N>` or `@ImageN`: a reserved label repeated replayed the references on H3.
    noun = "Picture" if style == "token" else "Image"
    if len(numbers) == 1:
        return f"{noun} {numbers[0]}"
    return f"{noun}s {', '.join(str(n) for n in numbers[:-1])} and {numbers[-1]}"


def _role_lines(
    name: str, roles: Sequence[str], count: int, first_position: int, style: str
) -> str:
    """Role sentences written from the allocation, so the numbers cannot drift from the refs."""
    grouped: dict[str, list[int]] = {}
    for offset, role in enumerate(roles[:count]):
        grouped.setdefault(role, []).append(first_position + offset)
    if set(grouped) <= {ROLE_FACE}:
        return ""
    out = ""
    for role in ROLES:
        numbers = grouped.get(role)
        if numbers:
            verb = "shows" if len(numbers) == 1 else "show"
            out += f" {_positions(style, numbers)} {verb} {name}'s {_ROLE_BINDINGS[role]}."
    return out


def prompt_prefix(
    name: str,
    description: str,
    roles: Sequence[str],
    *,
    first_position: int = 1,
    style: str = "ordinal",
    role_lines: bool = False,
    voice_position: int | None = None,
) -> str:
    """Text naming the positions a character lands on, in the form its model was trained on."""
    if style not in STYLES:
        raise ValueError(f"Unknown prompt style {style!r}. Use one of {', '.join(STYLES)}.")
    count = 0 if style == "description-only" else len(roles)
    if not count:
        # A LoRA carries the likeness, so the description is all the prompt needs.
        detail = " ".join(description.split())
        return f"{detail} " if detail else ""
    positions = [first_position + i for i in range(count)]
    if style == "token":
        tokens = " ".join(f"<Picture {n}>" for n in positions)
        plural = "" if len(positions) == 1 else "each"
        which = f"{tokens} {'shows' if not plural else 'show'}"
    elif style == "at-image":
        # Seedance's own addressing syntax, so each position is declared exactly once here.
        tokens = [f"@Image{n}" for n in positions]
        if len(tokens) == 1:
            which = f"{tokens[0]} shows"
        else:
            which = f"{', '.join(tokens[:-1])} and {tokens[-1]} show"
    elif len(positions) == 1:
        which = f"Image {positions[0]} shows"
    else:
        ordinals = [str(n) for n in positions]
        which = f"Images {', '.join(ordinals[:-1])} and {ordinals[-1]} show"
    line = f"{which} {name}, the same character in every image."
    if role_lines:
        line += _role_lines(name, roles, count, first_position, style)
    if voice_position is not None:
        line += voice_line(name, voice_position)
    detail = " ".join(description.split())
    if not detail:
        return f"{line} "
    # Prose, not comma tags: without this it runs into the user's prompt unpunctuated.
    if detail[-1] not in ".!?":
        detail += "."
    return f"{line} {detail} "
