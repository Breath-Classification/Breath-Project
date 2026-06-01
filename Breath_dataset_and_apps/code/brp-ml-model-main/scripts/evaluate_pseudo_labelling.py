import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from convert_tens_to_model_input import build_sequences

PROJECT_DIR = Path(__file__).resolve().parents[1]
PYTORCH_DIR = PROJECT_DIR / "models" / "Pytorch"
sys.path.append(str(PYTORCH_DIR))

DEFAULT_FILENAME = ""
DEFAULT_BASE_MODEL = ""
DEFAULT_OUTPUT = ""
STRATEGY = ""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pseudo labelling methods"
    )
    parser.add_argument("--filename", default=DEFAULT_FILENAME, type=Path)
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL, type=Path)
    parser.add_argument("--output_file", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--strategy", default=STRATEGY, type=str)
    
    return parser.parse_args()



def run_strategy(strategy): #decide_pseudo_labelling(strategy="XXx")
    _,decisions,relabeled = build_sequences()
    return decisions, relabeled


def visualize_decisions(): #keep =? drop =? relabeled =? 1->0 =? itd
    return

def save_sequences(): #seqence.txt  input for model
    return

def fine_tuning_for_strategies(): # FineTuning.py
    return

def evaluate():
    return

def compare_startegy_accuracy():
    return