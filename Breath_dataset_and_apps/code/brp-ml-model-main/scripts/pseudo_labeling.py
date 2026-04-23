from __future__ import annotations

from pathlib import Path
from typing import Literal


PseudoLabelDecision = Literal["keep", "drop", "relabeled"]

def threshold_per_user():
    return 

def breath_stability():
    return

def label_flip_rate():
    return

def average_length():
    return

def majority():
    return

def isloated():
    return

def slope():
    return

def confidence():
    return


def decide_pseudo_label(
    confidences: list[float] | None,
    position: int,
    confidence_threshold: float,
    label: int | None = None,
    time: float | None = None,
) -> PseudoLabelDecision:
    if confidences is None:
        return "keep"
    if confidences[position] < confidence_threshold:
        return "drop"
    return "keep"


def should_keep_pseudo_labeled(
    confidences: list[float] | None,
    position: int,
    confidence_threshold: float,
    label: int | None = None,
    time: float | None = None,
) -> bool:
    return decide_pseudo_label(confidences, position, confidence_threshold, label=label, time=time) != "drop"

def relabel (
    confidences: list[float] | None,
    position: int,
    confidence_threshold: float,
    label: int | None = None,
    time: float | None = None,
    strategy: str = "None",
)-> list[int]:
    
    if strategy == "majority":
        return majority()
    elif strategy ==  "isolated":
        return isloated()
    elif strategy == "slope":
        return slope()
    elif strategy == "confidence":
        return confidence()
    
    
