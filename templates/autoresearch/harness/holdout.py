"""Thresholdout — a reusable holdout with teeth (Dwork et al.; Blum-Hardt Ladder).

Nasty-simulation finding: adaptively keeping the best-on-dev over 80 noise-only candidates
inflated the dev score by +0.12 over the truth — the classic "overfit the eval by selecting on
it." The fix from the adaptive-data-analysis literature: don't read the holdout freely. Query it
through Thresholdout, which returns your dev number *unless* dev and holdout have diverged beyond
a noise threshold — in which case it spends a query and hands back the (honest, noised) holdout,
telling you you've overfit. Budget-limited, so the holdout stays informative across a whole climb.

Deterministic (seeded). This is `BEAT_IT.md`'s two-eval discipline made enforceable.
"""

from __future__ import annotations

import math
import random


def _laplace(rng: random.Random, scale: float) -> float:
    if scale <= 0:
        return 0.0
    u = rng.random() - 0.5
    return -scale * math.copysign(1, u) * math.log(1 - 2 * abs(u))


class Thresholdout:
    def __init__(self, threshold=0.02, noise=0.01, budget=20, seed=0):
        self.threshold = threshold
        self.noise = noise
        self.budget = budget
        self.rng = random.Random(seed)
        self._T = threshold + _laplace(self.rng, 2 * noise)

    def query(self, dev_stat: float, holdout_stat: float) -> dict:
        """Return what you're allowed to believe. If dev and holdout agree (within a noisy
        threshold) you get your dev number back for free. If they diverge — the signature of
        overfitting dev — you spend a query and get the honest (noised) holdout plus an
        `overfit` flag. When the budget is gone, the holdout is 'used up' and only dev is
        returned (with a warning) — stop trusting new dev gains."""
        if self.budget <= 0:
            return {"value": round(dev_stat, 4), "overfit": False, "budget": 0,
                    "note": "holdout budget exhausted — dev only; get fresh dev data"}
        if abs(dev_stat - holdout_stat) > self._T + _laplace(self.rng, 4 * self.noise):
            self.budget -= 1
            self._T = self.threshold + _laplace(self.rng, 2 * self.noise)   # fresh threshold
            return {"value": round(holdout_stat + _laplace(self.rng, self.noise), 4),
                    "overfit": True, "budget": self.budget,
                    "note": "dev and holdout diverged — revealing holdout; you've overfit dev"}
        return {"value": round(dev_stat, 4), "overfit": False, "budget": self.budget,
                "note": "dev and holdout agree within threshold"}
