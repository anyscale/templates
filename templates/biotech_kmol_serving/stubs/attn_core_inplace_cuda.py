"""Stub for the openfold CUDA kernel `attn_core_inplace_cuda`.

kMoL's `kmol/model/architectures/__init__.py` and `kmol/data/featurizers.py`
eagerly import the AlphaFold / openfold protein stack, which does a top-level
`import attn_core_inplace_cuda` (kmol/vendor/openfold/utils/kernel/attention_core.py).
That symbol is a compiled CUDA extension kMoL builds via setup.py — and building it
needs a GPU at build time.

The molecule GNN + ensemble serving path NEVER invokes openfold attention
(`attn_core_inplace_cuda.forward_/backward_` are only called inside openfold's
attention kernels). Placing this stub on PYTHONPATH satisfies the import so the
molecule path loads, without compiling anything. If the protein path is ever
actually exercised, these raise loudly rather than silently mis-computing.

This is dependency *provisioning*, not a modification of kMoL source.
"""


def forward_(*args, **kwargs):
    raise RuntimeError(
        "attn_core_inplace_cuda is stubbed in this molecule-serving image; "
        "the openfold/protein attention path is not available."
    )


def backward_(*args, **kwargs):
    raise RuntimeError(
        "attn_core_inplace_cuda is stubbed in this molecule-serving image; "
        "the openfold/protein attention path is not available."
    )
