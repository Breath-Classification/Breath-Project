from __future__ import annotations

from pathlib import Path


def resolve_txt_path(path: str | Path) -> Path:
    file_path = Path(path)
    if file_path.suffix != ".txt":
        file_path = file_path.with_suffix(".txt")
    return file_path


def remove_confidence(lines: list[str]) -> list[str]:
    new_lines: list[str] = []
    for line in lines:
        parts = [part.strip() for part in line.strip().split(",") if part.strip() != ""]
        if not parts:
            continue
        if len(parts) > 1:
            parts = parts[:-1]
        new_lines.append(",".join(parts) + "\n")
    return new_lines


def label(path: str | Path) -> None:
    file_path = resolve_txt_path(path)
    lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
    file_path.write_text("".join(remove_confidence(lines)), encoding="utf-8")
