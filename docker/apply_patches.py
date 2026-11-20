#!/usr/bin/env python3
"""Apply standard unified diff patches strictly without external `patch` or `git`."""

import re
import sys
from pathlib import Path

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def apply_file_diff(
    root: Path, rel_path: str, hunks: list[tuple[int, list[str]]]
) -> None:
    target = root / rel_path
    lines = target.read_text().splitlines(keepends=True)
    delta = 0

    for old_start, body in hunks:
        pos = max(0, old_start - 1 + delta)
        out_chunk: list[str] = []
        cursor = pos
        removed_or_context = 0
        added_or_context = 0

        for raw in body:
            if raw.startswith("\\"):
                continue
            tag, content = raw[0], raw[1:]
            if tag == " ":
                if cursor >= len(lines) or lines[cursor] != content:
                    got = lines[cursor] if cursor < len(lines) else "<EOF>"
                    raise ValueError(
                        f"{rel_path}:{cursor + 1}: context mismatch: expected {content!r}, got {got!r}"
                    )
                out_chunk.append(lines[cursor])
                cursor += 1
                removed_or_context += 1
                added_or_context += 1
            elif tag == "-":
                if cursor >= len(lines) or lines[cursor] != content:
                    got = lines[cursor] if cursor < len(lines) else "<EOF>"
                    raise ValueError(
                        f"{rel_path}:{cursor + 1}: removal mismatch: expected {content!r}, got {got!r}"
                    )
                cursor += 1
                removed_or_context += 1
            elif tag == "+":
                out_chunk.append(content)
                added_or_context += 1
            else:
                raise ValueError(f"{rel_path}: unsupported hunk line prefix {tag!r}")

        lines[pos : pos + removed_or_context] = out_chunk
        delta += added_or_context - removed_or_context

    target.write_text("".join(lines))


def apply_patch(root: Path, patch_path: Path) -> None:
    raw_lines = patch_path.read_text().splitlines(keepends=True)
    i = 0
    applied_files = 0

    while i < len(raw_lines):
        line = raw_lines[i]
        if not line.startswith("--- "):
            i += 1
            continue
        if i + 1 >= len(raw_lines) or not raw_lines[i + 1].startswith("+++ "):
            raise ValueError(f"{patch_path}: malformed file header at line {i + 1}")
        old_header = raw_lines[i][4:].strip().split("\t")[0]
        new_header = raw_lines[i + 1][4:].strip().split("\t")[0]
        rel = new_header[2:] if new_header.startswith("b/") else old_header[2:]
        i += 2

        hunks: list[tuple[int, list[str]]] = []
        while i < len(raw_lines) and raw_lines[i].startswith("@@ "):
            m = HUNK_RE.match(raw_lines[i])
            if not m:
                raise ValueError(
                    f"{patch_path}: malformed hunk header at line {i + 1}: {raw_lines[i]!r}"
                )
            old_start = int(m.group(1))
            i += 1
            body: list[str] = []
            while i < len(raw_lines) and not raw_lines[i].startswith(
                ("@@ ", "--- ", "diff ")
            ):
                body.append(raw_lines[i])
                i += 1
            hunks.append((old_start, body))

        apply_file_diff(root, rel, hunks)
        applied_files += 1

    print(f"Applied {patch_path.name} ({applied_files} files)")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("Usage: apply_patches.py <root_dir> <patch1> [<patch2> ...]")
    root = Path(sys.argv[1])
    for arg in sys.argv[2:]:
        apply_patch(root, Path(arg))


if __name__ == "__main__":
    main()
