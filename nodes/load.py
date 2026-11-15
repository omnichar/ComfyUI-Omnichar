# SPDX-License-Identifier: GPL-3.0-or-later
"""Load a ``.char`` and put it on a CHARACTER wire."""

from __future__ import annotations

import os

from omnichar_sdk import Character, CharError

from . import folders, licensing
from .common import CATEGORY, CHARACTER, licensed


class OmnicharLoadCharacter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "char": (
                    folders.listed(),
                    {"tooltip": "A .char file from models/characters."},
                ),
            },
            "optional": {
                "char_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Open a file by path instead of by name. It must sit inside a "
                            "registered character directory."
                        ),
                        "advanced": True,
                    },
                ),
                # Last, so a workflow saved before it existed keeps its values in place.
                "output": (
                    list(licensing.OUTPUTS),
                    {
                        "default": "image",
                        "tooltip": "What you'll generate. Only used for licensed characters.",
                        "advanced": True,
                    },
                ),
            },
        }

    RETURN_TYPES = (CHARACTER,)
    RETURN_NAMES = ("char",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Open an Omnichar Studio character file. Feed the result to the other Omnichar nodes to "
        "get its references, its description or its trained LoRA."
    )

    @classmethod
    def IS_CHANGED(cls, char, char_path="", output="image"):
        # One stat: hashing the archive costs over a second on a character carrying an adapter.
        try:
            path = folders.resolve(char, char_path)
            stat = os.stat(path)
        except (CharError, OSError):
            return float("nan")
        # Licensed: never cached, so every queued run is checked with Omnichar and reported once.
        if licensing.is_licensed(path):
            return float("nan")
        return (stat.st_mtime_ns, stat.st_size)

    @classmethod
    def VALIDATE_INPUTS(cls, char, char_path=""):
        # Through Character.open so the caps apply; this runs on every queue, before the node.
        try:
            path = folders.resolve(char, char_path)
            opened = Character.open(path)
        except CharError as error:
            return str(error)
        except Exception:
            return f"{char} is not a readable character file."
        # Offline only: a missing secret or an old SDK is known before anything runs.
        problem = licensing.precheck(opened, licensing.label(char, char_path, path))
        return True if problem is None else problem

    def load(self, char, char_path="", output="image"):
        path = folders.resolve(char, char_path)
        if not licensing.is_licensed(path):
            opened = Character.open(path)
            # Checked again on what was opened, so a file swapped in between is not let through.
            if licensed(opened) is None:
                return (opened,)
        name = licensing.label(char, char_path, path)
        return (licensing.unlock(licensing.open_cached(path), name, output),)
