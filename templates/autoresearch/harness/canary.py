"""R3 canaries + the leak audit — kill a doomed or too-good run before it costs budget.

`AUTORESEARCH.md` Iron Rule #7 / the leak-audit: the most dangerous result is one that's *too
good*, because it's usually a leak, not a discovery. These are cheap, mechanical detectors that
run on the numbers a job already produces. The headline is the shuffled-label control: if you
destroy the labels and the metric DOESN'T collapse to the prevalence floor, your pipeline is
reading the answer somewhere it shouldn't.
"""

from __future__ import annotations


def shuffled_label_control(shuffled_metric: float, prevalence_floor: float, tol=0.02) -> dict:
    """After randomly shuffling the labels, a healthy pipeline scores at the prevalence floor
    (AP≈base rate, accuracy≈majority). If it scores meaningfully above, the model is reading
    label information through a leak — the single most decisive leak test."""
    leaked = shuffled_metric > prevalence_floor + tol
    return {"leaked": leaked, "shuffled_metric": shuffled_metric, "floor": prevalence_floor,
            "verdict": "BROKEN" if leaked else "CLEAN",
            "note": ("shuffled-label metric is above the floor — target leak in the pipeline"
                     if leaked else "shuffled labels collapse to the floor — no obvious leak")}


def too_good_too_early(metric_step1: float, plausible_ceiling: float, margin=0.02) -> dict:
    """A near-ceiling decision metric at step 1 (before real learning) is the target-leak
    signature. Flags SUSPICIOUS; pair with the shuffled-label control to confirm."""
    suspicious = metric_step1 >= plausible_ceiling - margin
    return {"suspicious": suspicious, "verdict": "SUSPICIOUS" if suspicious else "CLEAN",
            "note": ("metric near ceiling at step 1 — likely a leak; run the shuffle control"
                     if suspicious else "step-1 metric is in a plausible range")}


def embedding_collapse(mean_pairwise_cosine: float, thresh=0.98) -> dict:
    """Representation collapse: if all embeddings point the same way (mean pairwise cosine →1),
    the encoder learned a constant. A universal training canary."""
    collapsed = mean_pairwise_cosine >= thresh
    return {"collapsed": collapsed, "verdict": "BROKEN" if collapsed else "CLEAN"}


def audit(shuffled=None, step1=None, ceiling=1.0, cosine=None) -> dict:
    """Roll the checks into one verdict: BROKEN (a control failed) > SUSPICIOUS (too-good) >
    CLEAN. Returns the worst-severity finding plus the individual results."""
    findings = {}
    if shuffled is not None:
        findings["shuffle"] = shuffled_label_control(shuffled["metric"], shuffled["floor"])
    if step1 is not None:
        findings["step1"] = too_good_too_early(step1, ceiling)
    if cosine is not None:
        findings["collapse"] = embedding_collapse(cosine)
    order = {"BROKEN": 2, "SUSPICIOUS": 1, "CLEAN": 0}
    worst = max((f["verdict"] for f in findings.values()), key=lambda v: order[v], default="CLEAN")
    return {"verdict": worst, "findings": findings}
