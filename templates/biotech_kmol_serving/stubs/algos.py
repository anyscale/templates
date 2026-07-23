"""Stub for the graphormer Cython `algos` module.

`kmol/data/featurizers.py` does a top-level `import algos` (line ~36). `algos` is a
compiled Cython module (shortest-path / spatial-encoding helpers) that kMoL builds
via setup.py for the Graphormer featurizer/collater.

The molecule ensemble serving path uses the plain `graph` featurizer + `general`
collater, which never touch `algos`. This stub satisfies the import so nothing needs
to be compiled; any actual use raises loudly.

This is dependency *provisioning*, not a modification of kMoL source.
"""


def __getattr__(name):
    raise RuntimeError(
        f"graphormer algos.{name} is stubbed in this molecule-serving image; "
        "the Graphormer featurizer/collater path is not available."
    )
