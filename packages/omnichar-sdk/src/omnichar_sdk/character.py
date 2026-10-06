# SPDX-License-Identifier: Apache-2.0
"""``Character``: one ``.char``, read a member at a time so a held handle cannot go stale."""

from __future__ import annotations

import io
import struct
import zipfile
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import charfile as cf
from . import limits
from .lora import Lora, measure_portability, read_safetensors_header_stream
from .prompt import prompt_prefix
from .references import Reference, fit_roles
from .sheet import reference_sheet, sheet_png
from .voice import Voice, voice_of

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

#: PNG signature plus the IHDR chunk: width and height are at a fixed offset in the first 24 bytes.
_PNG_HEADER_BYTES = 24
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class CharacterInfo:
    """Everything about a character that costs nothing to read."""

    name: str
    char_id: str
    description: str
    hints: list[str]
    ref_count: int
    ref_archs: list[str]
    lora_archs: list[str]
    format_version: int
    created_at: int
    modified_at: int
    size_bytes: int
    #: Seconds of stored voice, or None. Last and defaulted, so positional construction still works.
    voice_seconds: float | None = None


@contextmanager
def _memory_archive(data: bytes, label: str) -> Generator[zipfile.ZipFile]:
    """A zip held in memory, raising the same error a file on disk would."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data), "r")
    except zipfile.BadZipFile as error:
        raise cf.CharError(
            f"{label} is not a readable character file. It may have been truncated in "
            "transfer; download it again."
        ) from error
    try:
        yield archive
    finally:
        archive.close()


def _png_size(data: bytes) -> tuple[int, int]:
    """Width and height from a PNG's IHDR, or zeroes when the bytes are not a PNG."""
    if len(data) < _PNG_HEADER_BYTES or not data.startswith(_PNG_SIGNATURE):
        return (0, 0)
    width, height = struct.unpack(">II", data[16:24])
    return (int(width), int(height))


class Character:
    """A ``.char`` on disk or in memory. Construct with `open`, `from_bytes` or `from_file`."""

    def __init__(
        self,
        manifest: cf.Manifest,
        *,
        path: Path | None = None,
        data: bytes | None = None,
        label: str = "",
        manifest_sha: str = "",
        stamp: tuple[int, int] | None = None,
        size_bytes: int = 0,
    ) -> None:
        self._manifest = manifest
        self._path = path
        self._data = data
        self._label = label
        self._manifest_sha = manifest_sha
        self._stamp = stamp
        self._size_bytes = size_bytes

    # --- construction -----------------------------------------------------------------------

    @classmethod
    def open(cls, path: Path | str) -> Character:
        """Open a ``.char``, validating only its manifest."""
        target = Path(path)
        with cf.open_archive(target) as archive:
            raw = cf.read_manifest_bytes(archive, target.name)
            manifest = cf.parse_manifest(raw, target.name)
        stat = target.stat()
        return cls(
            manifest,
            path=target,
            label=target.name,
            manifest_sha=cf.sha256_bytes(raw),
            stamp=(stat.st_mtime_ns, stat.st_size),
            size_bytes=stat.st_size,
        )

    @classmethod
    def from_bytes(cls, data: bytes, label: str = "character") -> Character:
        """Open a ``.char`` already in memory, where nothing can change under it."""
        with _memory_archive(data, label) as archive:
            cf.check_archive(archive, label)
            raw = cf.read_manifest_bytes(archive, label)
            manifest = cf.parse_manifest(raw, label)
        return cls(
            manifest,
            data=data,
            label=label,
            manifest_sha=cf.sha256_bytes(raw),
            size_bytes=len(data),
        )

    @classmethod
    def from_file(cls, stream: Any, label: str = "character") -> Character:
        """Read a ``.char`` from a stream, bounded so an endless one cannot exhaust memory."""
        cap = limits.MAX_TOTAL_UNCOMPRESSED_BYTES
        data = stream.read(cap + 1)
        if len(data) > cap:
            raise cf.CharError(
                f"{label} is larger than {cap} bytes, which no character reaches. "
                "Do not open it; get the character from a source you trust."
            )
        return cls.from_bytes(data, label)

    # --- identity ---------------------------------------------------------------------------

    @property
    def manifest(self) -> cf.Manifest:
        return self._manifest

    @property
    def name(self) -> str:
        return self._manifest.name or Path(self._label).stem

    @property
    def char_id(self) -> str:
        return self._manifest.char_id

    @property
    def format_version(self) -> int:
        """What the file declared, not what this build reads."""
        return self._manifest.format_version

    @property
    def created_at(self) -> int:
        return self._manifest.created_at

    @property
    def modified_at(self) -> int:
        return self._manifest.modified_at

    # --- member access ----------------------------------------------------------------------

    @contextmanager
    def _archive(self) -> Generator[zipfile.ZipFile]:
        """An open archive, proven to still be the file this character was built from."""
        if self._data is not None:
            with _memory_archive(self._data, self._label) as archive:
                yield archive
            return
        assert self._path is not None
        stat = self._path.stat()
        current = (stat.st_mtime_ns, stat.st_size)
        with cf.open_archive(self._path) as archive:
            if current != self._stamp:
                raw = cf.read_manifest_bytes(archive, self._label)
                if cf.sha256_bytes(raw) != self._manifest_sha:
                    raise cf.CharChanged(
                        f"{self._label} was replaced while it was open. Open it again to read "
                        "the new version."
                    )
                self._stamp = current
            yield archive

    def _read(self, member: str) -> bytes:
        """One member's bytes, refusing a nested archive."""
        if not member:
            raise cf.CharError(
                f"{self._label} lists a file with no path in its manifest, so it cannot be read. "
                "Export the character again from Omnichar Studio."
            )
        with self._archive() as archive:
            try:
                data = archive.read(member)
            except KeyError:
                raise self._missing(member) from None
        return cf.check_member(member, data, self._label)

    def _missing(self, member: str) -> cf.CharError:
        return cf.CharError(
            f"{self._label} has no {member!r}, which its manifest lists. The character is "
            "incomplete; export it again from Omnichar Studio."
        )

    def _prefix(self, archive: zipfile.ZipFile, member: str, count: int) -> bytes:
        """The first ``count`` bytes of a member, checked and decompressed no further than that."""
        try:
            with archive.open(member, "r") as stream:
                data = stream.read(count)
        except KeyError:
            raise self._missing(member) from None
        return cf.check_member(member, data, self._label)

    # --- the facade -------------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        """The whole file, for a caller that wants to write it somewhere."""
        if self._data is not None:
            return self._data
        assert self._path is not None
        return self._path.read_bytes()

    def get_description(self) -> str:
        """The locked character description, or an empty string when the file carries none."""
        member = str(self._manifest.text.get("path") or "")
        if not member:
            return ""
        return self._read(member).decode("utf-8", errors="replace")

    def get_hints(self) -> list[str]:
        return [str(hint) for hint in self._manifest.hints]

    def get_ref_archs(self) -> list[str]:
        """Architectures with a compiled reference set."""
        return sorted(k for k in self._manifest.payloads if not k.endswith(f"-{cf.PAYLOAD_LORA}"))

    def get_lora_archs(self) -> list[str]:
        """Architectures with a trained adapter."""
        suffix = f"-{cf.PAYLOAD_LORA}"
        return sorted(k[: -len(suffix)] for k in self._manifest.payloads if k.endswith(suffix))

    def get_info(self) -> CharacterInfo:
        return CharacterInfo(
            name=self.name,
            char_id=self.char_id,
            description=self.get_description(),
            hints=self.get_hints(),
            ref_count=len(self._manifest.refs),
            ref_archs=self.get_ref_archs(),
            lora_archs=self.get_lora_archs(),
            format_version=self.format_version,
            created_at=self.created_at,
            modified_at=self.modified_at,
            size_bytes=self._size_bytes,
            voice_seconds=voice.seconds if (voice := self.get_voice()) else None,
        )

    def get_voice(self) -> Voice | None:
        """The character's voice, or None. Only MiniMax H3 Reference to Video reads one."""
        return voice_of(self._manifest, self._read_voice)

    def _read_voice(self, member: str) -> bytes:
        """A voice member, its size checked before a byte is decompressed."""
        from .voice import MAX_PAYLOAD_BYTES, MAX_SAMPLE_BYTES

        cap = MAX_PAYLOAD_BYTES if member.startswith("voice/payload/") else MAX_SAMPLE_BYTES
        with self._archive() as archive:
            try:
                size = archive.getinfo(member).file_size
            except KeyError:
                raise self._missing(member) from None
        if size > cap:
            raise cf.CharError(
                f"{self._label}'s voice file {member!r} is {size} bytes, over the {cap} a voice "
                "can be. Do not use it; get the character from a source you trust."
            )
        return self._read(member)

    def get_references(
        self,
        arch: str | None = None,
        role: str | None = None,
        origin: str | None = None,
        limit: int | None = None,
    ) -> list[Reference]:
        """References in manifest order, which is what a prompt's positions refer to."""
        if arch is not None and origin is not None:
            raise ValueError(
                "get_references takes arch or origin, not both. A compiled payload records no "
                f"per-reference origin, so origin={origin!r} cannot be applied to arch={arch!r}. "
                "Drop origin, or drop arch to read the original references."
            )
        if arch is None:
            entries = list(self._manifest.refs)
        else:
            payload = self._manifest.payloads.get(arch)
            if not isinstance(payload, dict):
                raise cf.CharError(
                    f"{self.name} has no compiled references for {arch}. It has "
                    f"{', '.join(self.get_ref_archs()) or 'none'}."
                )
            entries = list(payload.get("files") or [])

        refs = self._build_references(entries)
        if origin is not None:
            refs = [ref for ref in refs if ref.origin == origin]
        if role is not None:
            refs = [ref for ref in refs if ref.role == role]
        if limit is not None:
            refs = fit_roles(refs, limit)
        return refs

    def _build_references(self, entries: Sequence[dict[str, Any]]) -> list[Reference]:
        """Reference objects, reading a PNG header only for entries that record no size."""
        missing = [
            str(e.get("path") or "")
            for e in entries
            if not (int(e.get("width") or 0) and int(e.get("height") or 0))
        ]
        sizes: dict[str, tuple[int, int]] = {}
        if missing:
            with self._archive() as archive:
                for member in missing:
                    if not member:
                        continue
                    sizes[member] = _png_size(
                        self._prefix(archive, member, _PNG_HEADER_BYTES)
                    )
        out: list[Reference] = []
        for entry in entries:
            path = str(entry.get("path") or "")
            width = int(entry.get("width") or 0)
            height = int(entry.get("height") or 0)
            if not (width and height):
                width, height = sizes.get(path, (0, 0))
            out.append(
                Reference(
                    path=path,
                    role=cf.role_of(entry),
                    origin=cf.origin_of(entry),
                    width=width,
                    height=height,
                    sha256=str(entry.get("sha256") or ""),
                    _read=self._read,
                )
            )
        return out

    def get_lora(self, arch: str | None = None) -> Lora | None:
        """The adapter for ``arch``; with several and no ``arch`` it raises rather than guess."""
        available = self.get_lora_archs()
        if not available:
            return None
        if arch is None:
            if len(available) > 1:
                raise cf.CharError(
                    f"{self.name} carries adapters for {', '.join(available)}. "
                    "Name one with get_lora(arch=...)."
                )
            arch = available[0]
        key = cf.payload_key(arch, cf.PAYLOAD_LORA)
        entry = self._manifest.payloads.get(key)
        if not isinstance(entry, dict):
            return None
        files = [str(f.get("path")) for f in entry.get("files") or [] if f.get("path")]
        if not files:
            return None
        member = files[0]
        raw_training = entry.get("training")
        training: dict[str, Any] = raw_training if isinstance(raw_training, dict) else {}
        filename = Path(member).name
        keys = self._adapter_keys(member, filename)
        verdict, reason = measure_portability(arch, keys)
        return Lora(
            arch=arch,
            base=str(entry.get("base") or ""),
            rank=int(training.get("rank") or 0),
            steps=int(training.get("steps") or 0),
            resolution=int(training.get("resolution") or 0),
            strength=float(entry.get("strength") or 1.0),
            filename=filename,
            portability=verdict,
            reason=reason,
            _read=self._read,
            _read_header=lambda: self._adapter_header(member, filename),
            _member=member,
        )

    def _adapter_keys(self, member: str, label: str) -> list[str]:
        """Tensor names from the adapter's header."""
        return [k for k in self._adapter_header(member, label) if k != "__metadata__"]

    def _adapter_header(self, member: str, label: str) -> dict[str, Any]:
        """The adapter's header, decompressing no more of the member than the header itself."""
        with self._archive() as archive:
            # The nested-archive guard, before the header is streamed out of the member.
            self._prefix(archive, member, len(limits.ZIP_MAGIC))
            try:
                with archive.open(member, "r") as stream:
                    header = read_safetensors_header_stream(stream, label)
            except KeyError:
                raise self._missing(member) from None
        return header

    def reference_sheet(
        self,
        arch: str | None = None,
        role: str | None = None,
        limit: int | None = None,
        origin: str | None = None,
        *,
        first_position: int = 1,
        **options: Any,
    ) -> Image:
        """References as one contact sheet, numbered from ``first_position`` as a prompt is."""
        refs = self.get_references(arch=arch, role=role, origin=origin, limit=limit)
        options.setdefault("title", self.name)
        options.setdefault("subtitle", f"{len(refs)} references" + (f" · {arch}" if arch else ""))
        return reference_sheet(refs, first_position=first_position, **options)

    def reference_sheet_png(self, *args: Any, **options: Any) -> bytes:
        """The contact sheet as PNG bytes."""
        return sheet_png(self.reference_sheet(*args, **options))

    def save_reference_sheet(self, path: Path | str, *args: Any, **options: Any) -> Path:
        """Write the contact sheet to ``path`` as a PNG, adding the extension when it has none."""
        target = Path(path)
        if target.suffix.lower() != ".png":
            target = target.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.reference_sheet_png(*args, **options))
        return target

    def get_prompt(
        self,
        style: str = "ordinal",
        first_position: int = 1,
        role_lines: bool = False,
        count: int | None = None,
        arch: str | None = None,
        voice_position: int | None = None,
    ) -> str:
        """Prompt text naming the positions this character's references will occupy."""
        refs = self.get_references(arch=arch, limit=count)
        return prompt_prefix(
            self.name,
            self.get_description(),
            [ref.role for ref in refs],
            first_position=first_position,
            style=style,
            role_lines=role_lines,
            voice_position=voice_position if self.get_voice() else None,
        )

    def __repr__(self) -> str:
        return f"<Character {self.name!r} refs={len(self._manifest.refs)}>"
