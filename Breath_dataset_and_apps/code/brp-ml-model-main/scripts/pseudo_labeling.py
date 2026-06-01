from __future__ import annotations

from collections import Counter
from typing import Literal

PseudoLabelDecision = Literal["keep", "drop"] | tuple[Literal["relabeled"], int]
SLOPE = 0.02


# Statistic functions
def label_flip_rate(path: str, labels: list[int] | None) -> int:
    if not labels:
        return 0

    transitions = 0
    for i in range(1, len(labels)):
        if labels[i - 1] != labels[i]:
            transitions += 1
    return transitions


def average_length(path: str, labels: list[int] | None, class_p: int) -> float:
    if not labels:
        return 0.0

    counter = 1
    class_sum = 0
    class_count = 0
    for i in range(1, len(labels)):
        if labels[i - 1] == labels[i] and labels[i] == class_p:
            counter += 1
        else:
            if labels[i - 1] == class_p:
                class_sum += counter
                class_count += 1
            counter = 1

    if labels[-1] == class_p:
        class_sum += counter
        class_count += 1

    return class_sum / class_count if class_count else 0.0


def threshold_per_user():
    return


def breath_stability():
    return


# Helper functions
def _convert_labels(
    labels: list[int],
    window_size: int,
    position: int,
) -> list[int]:
    converted_labels = []
    for i in range(window_size // 2, len(labels), window_size):
        converted_labels.append(labels[i])
    return converted_labels


def _window_bounds(position: int, window_size: int, length: int) -> tuple[int, int]:
    half_window = max(1, window_size // 2)
    start = max(0, position - half_window)
    end = min(length, position + half_window + 1)
    return start, end


def _local_slope(
    values: list[float],
    position: int,
    window_size: int,
    times: list[float] | None = None,
) -> float:
    start, end = _window_bounds(position, window_size, len(values))
    if end - start < 2:
        return 0.0

    delta_value = values[end - 1] - values[start]
    if times is not None and len(times) >= end:
        delta_time = times[end - 1] - times[start]
        if delta_time == 0:
            return 0.0
        return delta_value / delta_time

    return delta_value / (end - start - 1)


# Labeling
def majority(
    position: int,
    labels: list[int] | None,
    window_size: int,
) -> PseudoLabelDecision:
    if not labels or position < 0 or position >= len(labels):
        return "keep"

    start, end = _window_bounds(position, window_size, len(labels))
    major = Counter(labels[start:end]).most_common(1)[0][0]
    if major == labels[position]:
        return "keep"
    return "relabeled", major


def isolated(
    position: int,
    labels: list[int] | None,
    window_size: int,
) -> PseudoLabelDecision:
    if not labels or position <= 0 or position >= len(labels) - 1:
        return "keep"

    if labels[position] != labels[position - 1] and labels[position] != labels[position + 1]:
        return "relabeled", labels[position - 1]
    return "keep"


def relabel_by_physical_rules(
    position: int,
    labels: list[int] | None,
    times: list[float] | None,
    window_size: int,
    amplitude: float = 0.0,
    values: list[float] | None = None,
) -> PseudoLabelDecision:
    if not labels or position <= 0 or position >= len(labels) - 1:
        return "keep"

    left = labels[position - 1]
    right = labels[position + 1]
    current = labels[position]

    if current != left and current != right:
        if left == right:
            return "relabeled", left
        return "relabeled", Counter([left, right]).most_common(1)[0][0]

    if values is not None and amplitude > 0:
        return slope(position, labels, times, window_size, amplitude, values)

    return "keep"


def slope(
    position: int,
    labels: list[int] | None,
    times: list[float] | None,
    window_size: int,
    amplitude: float,
    values: list[float],
    slope_threshold: float = SLOPE,
) -> PseudoLabelDecision:
    if not labels or not values:
        return "keep"
    if position < 0 or position >= len(labels) or position >= len(values):
        return "keep"

    local_slope = _local_slope(values, position, window_size, times)
    if abs(local_slope) < slope_threshold or amplitude <= 0:
        return "keep"

    if local_slope > 0 and position + 1 < len(labels):
        target = labels[position + 1]
        if target != labels[position]:
            return "relabeled", target
    elif local_slope < 0 and position - 1 >= 0:
        target = labels[position - 1]
        if target != labels[position]:
            return "relabeled", target

    return "keep"


def cut(
    confidences: list[float] | None,
    position: int,
    confidence_threshold: float,
) -> PseudoLabelDecision:
    if confidences is None:
        return "keep"
    if position < 0 or position >= len(confidences):
        return "drop"
    if confidences[position] < confidence_threshold:
        return "drop"
    return "keep"

#~Labeling 



# Main strategy
def relabel(
    confidences: list[float] | None,
    position: int,
    confidence_threshold: float,
    labels: list[int] | None,
    times: list[float] | None,
    window_size: int,
    amplitude: float,
    values: list[float],
    strategy: str = "none",
) -> PseudoLabelDecision:
    if strategy == "majority":
        return majority(position, labels, window_size)
    if strategy == "isolated":
        return isolated(position, labels, window_size)
    if strategy == "slope":
        return slope(position, labels, times, window_size, amplitude, values)
    if strategy == "cut":
        return cut(confidences, position, confidence_threshold)
    if strategy == "physical":
        return relabel_by_physical_rules(position, labels, times, window_size, amplitude, values)
    return "keep"


def decide_pseudo_label(
    confidences: list[float] | None,
    position: int,
    confidence_threshold: float,
    labels: list[int] | None,
    times: list[float] | None,
    window_size: int,
    amplitude: float,
    values: list[float],
) -> PseudoLabelDecision:
    if confidences is None:
        return "keep"
    return relabel(
        confidences,
        position,
        confidence_threshold,
        labels,
        times,
        window_size,
        amplitude,
        values,
        strategy="physical",
    )
#~Main Strategy


    decision = relabel(
        confidences,
        position,
        confidence_threshold,
        labels,
        times,
        window_size,
        amplitude,
        values,
        strategy="physical",
    )
    if decision != "keep":
        return decision

    return cut(confidences, position, confidence_threshold)
