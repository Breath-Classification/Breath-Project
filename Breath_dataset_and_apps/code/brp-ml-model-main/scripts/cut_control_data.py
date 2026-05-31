import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
PYTORCH_DIR = PROJECT_DIR / "models" / "Pytorch"
sys.path.append(str(PYTORCH_DIR))

DEFAULT_FILENAME = ""
DEFAULT_OUTPUT = ""
DEFAULT_LINES = 300
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Cut control data fine-tuning of recording"
    )
    parser.add_argument("--filename",default=DEFAULT_FILENAME, type=Path)
    parser.add_argument("--output", default= DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--lines", default=DEFAULT_LINES, type=int)
    
    return parser.parse_args()

def cut_lines(filename: Path, output: Path, lines_to_remove: int):
    lines = filename.read_text(encoding="utf8").splitlines(keepends=True)

    header = lines[:1]
    data = lines[1:]

    output.write_text("".join(header + data[lines_to_remove:]), encoding="utf8")

def main():
    args = parse_args()
    cut_lines(args.filename, args.output, args.lines)

if __name__ == "__main__":
    main()
            