"""Family-wise / FDR control — so a campaign of many experiments doesn't manufacture a win.

Nasty-simulation finding: run 40 NULL experiments and, at a per-test 95% noise floor, ~1-2
clear it by pure luck. A campaign runs *many* experiments, so per-test significance is not
enough — you must control error across the whole family. Benjamini-Hochberg (FDR) is the right
tool for research screening (it controls the *expected fraction of false discoveries* among the
ones you keep, and is far less brutal than Bonferroni when you have many real effects too).

Usage: collect a one-sided `metrics.p_value_gain` per experiment, then `benjamini_hochberg`
tells you which survive at a chosen false-discovery rate `q`.
"""

from __future__ import annotations


def benjamini_hochberg(pvals, q=0.05) -> list:
    """Benjamini-Hochberg step-up. Returns a list of booleans (True = significant after FDR
    control at level `q`), aligned to the input order. Controls the expected proportion of
    false positives among the rejections — the correct screen for a research backlog."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])          # ascending p
    max_rank = 0
    for rank, i in enumerate(order, 1):
        if pvals[i] <= q * rank / m:
            max_rank = rank                                    # largest rank meeting the line
    rejected = [False] * m
    for rank, i in enumerate(order, 1):
        if rank <= max_rank:
            rejected[i] = True
    return rejected


def bonferroni(pvals, alpha=0.05) -> list:
    """Bonferroni — control the family-wise error rate (any false positive). Stricter than FDR;
    use when even one false 'win' shipping is unacceptable (a headline claim), not for screening."""
    m = len(pvals)
    return [p <= alpha / m for p in pvals] if m else []


def expected_false_wins(n_tests, alpha=0.05) -> float:
    """The napkin math that motivates the above: at per-test level `alpha`, this many of
    `n_tests` NULL experiments are expected to look 'significant' by chance."""
    return round(n_tests * alpha, 2)
