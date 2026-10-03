import numpy as np
from scipy.stats import wilcoxon, t, rankdata
from statsmodels.stats.multitest import multipletests


# ============================================================
# PROPOSED MODEL
# ============================================================

proposed = {
    "test_acc": [
        0.9789937106918238,
        0.9177358490566038,
        0.9603703703703703,
        0.9700000000000000,
        0.9494117647058824,
        0.9739869281045751,
        0.9809090909090908,
        0.9745833333333334,
    ],

    "epsilon_accuracy": [
        0.9789937106918238,
        0.9177358490566038,
        0.9603703703703703,
        0.9700000000000000,
        0.9494117647058824,
        0.9739869281045751,
        0.9809090909090908,
        0.9745833333333334,
    ],

    "transition_accuracy": [
        0.9316294970176303,
        0.7760331056238219,
        0.9113314624636777,
        0.9021028853940246,
        0.9308314096862069,
        0.9959795596027929,
        1.0130321614834847,  # WARNING: > 1
        0.9687113372536910,
    ],

    "cycle_accuracy": [
        0.8530612244897959,
        0.7361702127659575,
        0.9054545454545455,
        0.8379310344827585,
        0.8135593220338982,
        0.8642857142857142,
        0.9571428571428571,
        0.9546666666666667,
    ],

    "transition_edtt_f1": [
        87.44588438735177,
        69.66056053855861,
        87.30759491785453,
        83.35506865225399,
        81.10053807835223,
        83.32472488506576,
        86.79246668992928,
        93.1155816041929,
    ],

    "transition_timing_mae": [
        0.488347659400291,
        0.28098723626162003,
        0.20503883845910673,
        0.33766108649102833,
        0.26949825388026605,
        0.29318524130963863,
        0.3005884753921776,
        0.26914672115333765,
    ],
}


# ============================================================
# BASELINE LSTM
# ============================================================
'''
baseline = {
    "test_acc": [
        0.9840153452685423,
        0.897439180537772,
        0.9293675641828429,
        0.938978978978979,
        0.9388743455497381,
        0.972372769332452,
        0.9744418783679754,
        0.9681016231474947,
    ],

    "epsilon_accuracy": [
        0.9840153452685423,
        0.897439180537772,
        0.9293675641828429,
        0.938978978978979,
        0.9388743455497381,
        0.972372769332452,
        0.9744418783679754,
        0.9681016231474947,
    ],

    "transition_accuracy": [
        0.8533393707112529,
        0.6484559172477509,
        0.9315775579674133,
        0.7509109798795859,
        0.842151958410023,
        0.9694990929922437,
        0.9724673113806963,
        0.9382447263718939,
    ],

    "cycle_accuracy": [
        0.8212765957446809,
        0.6391304347826087,
        0.8444444444444444,
        0.7403508771929825,
        0.7049180327868853,
        0.8350877192982455,
        0.9304347826086957,
        0.9135135135135135,
    ],
}
'''
baseline = {
    "test_acc": [
        0.9827672955974842,
        0.9230188679245284,
        0.9619753086419752,
        0.9676190476190476,
        0.9515032679738562,
        0.9708496732026143,
        0.9798484848484849,
        0.97625,
    ],

    "epsilon_accuracy": [
        0.9827672955974842,
        0.9230188679245284,
        0.9619753086419752,
        0.9676190476190476,
        0.9515032679738562,
        0.9708496732026143,
        0.9798484848484849,
        0.97625,
    ],

    "transition_edtt_f1": [
        87.44588438735177,
        69.66056053855861,
        87.30759491785453,
        83.35506865225399,
        81.10053807835223,
        84.32472488506576,
        86.79246668992928,
        93.1155816041929,
    ],

    "transition_timing_mae": [
        0.388347659400291,
        0.28098723626162003,
        0.20503883845910673,
        0.33766108649102833,
        0.26949825388026605,
        0.29318524130963863,
        0.3005884753921776,
        0.26914672115333765,
    ],

    "cycle_accuracy": [
        0.8530612244897959,
        0.727659574468085,
        0.9090909090909092,
        0.8310344827586207,
        0.8440677966101695,
        0.8428571428571429,
        0.9514285714285714,
        0.9466666666666667,
    ],
}



# ============================================================
# SETTINGS
# ============================================================

ALPHA = 0.05

metrics = [
    "test_acc",
    "epsilon_accuracy",
    "transition_edtt_f1",
    "cycle_accuracy",
    "transition_timing_mae"
]


metric_names = {
    "test_acc": "Standard Accuracy",
    "epsilon_accuracy": "Epsilon Accuracy",
    "cycle_accuracy": "Cycle Accuracy",
    "transition_edtt_f1" :"Transition EDDT",
    "transition_timing_mae": "Transition MAE",
}


# ============================================================
# CHECK DATA
# ============================================================

for metric in metrics:

    if metric not in proposed:
        raise ValueError(f"Missing metric in proposed: {metric}")

    if metric not in baseline:
        raise ValueError(f"Missing metric in baseline: {metric}")

    if len(proposed[metric]) != len(baseline[metric]):
        raise ValueError(
            f"{metric}: Proposed and baseline must have "
            f"the same number of participants."
        )

    if len(proposed[metric]) < 2:
        raise ValueError(
            f"{metric}: At least two participants are required."
        )


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(values):
    """
    Calculate:
        mean
        sample standard deviation
        95% confidence interval
    """

    values = np.asarray(values, dtype=float)

    n = len(values)

    mean = np.mean(values)

    sd = np.std(
        values,
        ddof=1
    )

    standard_error = sd / np.sqrt(n)

    confidence_low, confidence_high = t.interval(
        confidence=0.95,
        df=n - 1,
        loc=mean,
        scale=standard_error,
    )

    return (
        mean,
        sd,
        confidence_low,
        confidence_high,
    )


# ============================================================
# EFFECT SIZE
# ============================================================

def calculate_effect_size_r(proposed_values, baseline_values):
    """
    Effect size for Wilcoxon signed-rank test.

    r = |Z| / sqrt(N)

    Interpretation:
        ~0.1 = small
        ~0.3 = medium
        ~0.5 = large
    """

    proposed_values = np.asarray(
        proposed_values,
        dtype=float
    )

    baseline_values = np.asarray(
        baseline_values,
        dtype=float
    )

    differences = proposed_values - baseline_values

    # Remove zero differences
    differences = differences[differences != 0]

    n = len(differences)

    if n == 0:
        return np.nan

    result = wilcoxon(
        proposed_values,
        baseline_values,
        alternative="two-sided",
        method="approx",
    )

    z = result.zstatistic

    r = abs(z) / np.sqrt(n)

    return r


def calculate_rank_biserial_correlation(proposed_values, baseline_values):
    """
    Rank-biserial correlation for the Wilcoxon signed-rank test.

    r_rb = (W+ - W-) / (W+ + W-)

    Interpretation:
        r_rb > 0  -> proposed model tends to have higher values
        r_rb < 0  -> baseline model tends to have higher values
        r_rb = 0  -> no directional difference
    """

    proposed_values = np.asarray(
        proposed_values,
        dtype=float
    )

    baseline_values = np.asarray(
        baseline_values,
        dtype=float
    )

    differences = proposed_values - baseline_values

    # Remove zero differences
    differences = differences[differences != 0]

    if len(differences) == 0:
        return np.nan

    # Calculate absolute differences and their ranks
    absolute_differences = np.abs(differences)

    ranks = rankdata(
        absolute_differences,
        method="average"
    )

    # Sum ranks according to the direction of the difference
    positive_ranks = np.sum(
        ranks[differences > 0]
    )

    negative_ranks = np.sum(
        ranks[differences < 0]
    )

    # Rank-biserial correlation
    r_rb = (
        positive_ranks - negative_ranks
    ) / (
        positive_ranks + negative_ranks
    )

    return r_rb




# ============================================================
# WILCOXON TESTS
# ============================================================

results = []
p_values = []


for metric in metrics:

    proposed_values = np.asarray(
        proposed[metric],
        dtype=float
    )

    baseline_values = np.asarray(
        baseline[metric],
        dtype=float
    )

    # --------------------------------------------------------
    # Descriptive statistics
    # --------------------------------------------------------

    proposed_mean, proposed_sd, proposed_ci_low, proposed_ci_high = (
        calculate_statistics(proposed_values)
    )

    baseline_mean, baseline_sd, baseline_ci_low, baseline_ci_high = (
        calculate_statistics(baseline_values)
    )

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    differences = (
        proposed_values - baseline_values
    )

    mean_difference = np.mean(differences)

    median_difference = np.median(differences)

    # --------------------------------------------------------
    # Wilcoxon signed-rank test
    # --------------------------------------------------------

    statistic, p_value = wilcoxon(
        proposed_values,
        baseline_values,
        alternative="two-sided",
        method="auto",
    )

    # --------------------------------------------------------
    # Effect size
    # --------------------------------------------------------
    effect_size_biserial = calculate_rank_biserial_correlation(proposed_values,baseline_values)
    effect_size = calculate_effect_size_r(
        proposed_values,
        baseline_values,
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "metric": metric,
        "proposed_mean": proposed_mean,
        "proposed_sd": proposed_sd,
        "proposed_ci_low": proposed_ci_low,
        "proposed_ci_high": proposed_ci_high,

        "baseline_mean": baseline_mean,
        "baseline_sd": baseline_sd,
        "baseline_ci_low": baseline_ci_low,
        "baseline_ci_high": baseline_ci_high,

        "mean_difference": mean_difference,
        "median_difference": median_difference,

        "W": statistic,
        "p": p_value,

        "effect_size_r": effect_size,
        "effect_size_biserial": effect_size_biserial,
    })

    p_values.append(p_value)


# ============================================================
# HOLM CORRECTION
# ============================================================

reject, corrected_p_values, _, _ = multipletests(
    p_values,
    alpha=ALPHA,
    method="holm",
)


for result, corrected_p, is_significant in zip(
    results,
    corrected_p_values,
    reject,
):

    result["p_corrected"] = corrected_p

    result["significant"] = bool(is_significant)


# ============================================================
# PRINT DETAILED RESULTS
# ============================================================

print()
print("=" * 100)
print("STATISTICAL COMPARISON")
print("PROPOSED MODEL vs BASELINE LSTM")
print("=" * 100)


for result in results:

    metric = result["metric"]

    print()
    print("-" * 100)
    print(metric_names[metric])
    print("-" * 100)

    print(
        f"Proposed:"
        f"  {result['proposed_mean']:.4f}"
        f" ± {result['proposed_sd']:.4f}"
    )

    print(
        f"Proposed 95% CI:"
        f"  [{result['proposed_ci_low']:.4f}, "
        f"{result['proposed_ci_high']:.4f}]"
    )

    print()

    print(
        f"Baseline LSTM:"
        f"  {result['baseline_mean']:.4f}"
        f" ± {result['baseline_sd']:.4f}"
    )

    print(
        f"Baseline 95% CI:"
        f"  [{result['baseline_ci_low']:.4f}, "
        f"{result['baseline_ci_high']:.4f}]"
    )

    print()

    print(
        f"Mean difference:"
        f"  {result['mean_difference']:+.4f}"
    )

    print(
        f"Median difference:"
        f"  {result['median_difference']:+.4f}"
    )

    print()

    print(
        f"Wilcoxon W:"
        f"  {result['W']:.4f}"
    )

    print(
        f"p-value:"
        f"  {result['p']:.6f}"
    )

    print(
        f"Holm-corrected p-value:"
        f"  {result['p_corrected']:.6f}"
    )

    print(
        f"Effect size r:"
        f"  {result['effect_size_r']:.4f}"
    )

    print(
        f"Effect size biserial:"
        f"  {result['effect_size_biserial']:.4f}"
    )

    print(
        f"Statistically significant:"
        f"  {'YES' if result['significant'] else 'NO'}"
    )


# ============================================================
# SUMMARY TABLE
# ============================================================

print()
print()
print("=" * 120)
print("SUMMARY")
print("=" * 120)

header = (
    f"{'Metric':<25}"
    f"{'Proposed':>15}"
    f"{'Baseline':>15}"
    f"{'Difference':>15}"
    f"{'p':>12}"
    f"{'p Holm':>12}"
    f"{'r':>10}"
    f"{'Sig.':>8}"
)

print(header)
print("-" * 120)


for result in results:

    proposed_text = (
        f"{result['proposed_mean']:.3f} "
        f"± {result['proposed_sd']:.3f}"
    )

    baseline_text = (
        f"{result['baseline_mean']:.3f} "
        f"± {result['baseline_sd']:.3f}"
    )

    print(
    f"{metric_names[result['metric']]:<25}"
    f"{proposed_text:>15}"
    f"{baseline_text:>15}"
    f"{result['mean_difference']:>+15.3f}"
    f"{result['p']:>12.6f}"
    f"{result['p_corrected']:>12.6f}"
    f"{result['effect_size_r']:>10.3f}"
    f"{result['effect_size_biserial']:>10.3f}"
    f"{'YES' if result['significant'] else 'NO':>8}"
)
    


print("=" * 120)
print()
print(f"Significance level: α = {ALPHA}")
print("Test: paired Wilcoxon signed-rank test")
print("Multiple comparisons correction: Holm")
print()