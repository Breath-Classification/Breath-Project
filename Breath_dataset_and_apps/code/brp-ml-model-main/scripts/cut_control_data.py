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
CONTROL_DATA = False
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Cut control data fine-tuning of recording"
    )
    parser.add_argument("--filename",default=DEFAULT_FILENAME, type=Path)
    parser.add_argument("--output", default= DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--lines", default=DEFAULT_LINES, type=int)
    parser.add_argument("--control_data", default=CONTROL_DATA, type=bool)
    
    return parser.parse_args()

def cut_lines(filename: Path, output: Path, lines: int, control_data: bool):
    file = filename.read_text(encoding="utf8").splitlines(keepends=True)

    header = file[:1]
    data = file[1:]
    if not control_data:
        output.write_text("".join(header + data[lines:]), encoding="utf8")
    else:
        output.write_text("".join(header + data[:lines]), encoding="utf8")

def main():
    args = parse_args()
    cut_lines(args.filename, args.output, args.lines, args.control_data)

if __name__ == "__main__":
    main()
            