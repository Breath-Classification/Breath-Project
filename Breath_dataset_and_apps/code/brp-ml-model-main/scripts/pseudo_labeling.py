from __future__ import annotations

from pathlib import Path
from typing import Literal
from collections import Counter

PseudoLabelDecision = Literal["keep", "drop", "relabeled"]

def threshold_per_user():
    return 

def breath_stability():
    return

def label_flip_rate():
    return

def average_length():
    return

def majority(
    position: int,
    labels: list[int] | None,
    window_size: int,):
    
    start = position - window_size//2
    end = position + window_size//2
    
    preds = []
    for i in range(start,end):
        preds.append(labels[i])
    
    major = Counter(preds).most_common(1)[0][0]
    
    if major == labels[position]:
        return "keep"
    else:
        return "relabeled", major

def isloated():
    return

def slope():
    return

def confidence():
    return

def cut(confidences: list[float] | None,
    position: int,
    confidence_threshold: float,):
    if confidences[position] < confidence_threshold:
        return "drop"
    return "None"


def decide_pseudo_label(
    confidences: list[float] | None,
    position: int,
    confidence_threshold: float,
    labels: list[int] | None,
    times: list[float] | None,
    window_size: int,
    amplitude:float,
    values: list[float],
) -> PseudoLabelDecision:
    if confidences is None:
        return "keep"
    if confidences[position] < confidence_threshold:
        return "drop"
    return "keep"


def relabel (
    confidences: list[float] | None,
    position: int,
    confidence_threshold: float,
    labels: list[int] | None,
    times: list[float] | None,
    window_size: int,
    amplitude:float,
    values: list[float],
    strategy: str = "None",
    
)-> list[int]:
    
    if strategy == "majority":
        return majority(position,labels,window_size) #TODO think about bigger window_size
    elif strategy ==  "isolated":
        return isloated()
    elif strategy == "slope":
        return slope()
    elif strategy == "confidence":
        return confidence()
    elif strategy == "cut":
        return cut(confidences,position,confidence_threshold)
    
    
