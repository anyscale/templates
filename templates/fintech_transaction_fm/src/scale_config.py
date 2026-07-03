"""Per-scale pipeline settings: one YAML per scale under configs/.

``configs/<scale>.yaml`` is the single source of truth for everything that
differs between mini/small/full — data sampling, tokenization, model dims,
training, and embedding extraction. To run a custom experiment:

    cp configs/mini.yaml configs/ttest.yaml    # edit it
    python scripts/run_pipeline.py --scale ttest

Any ``configs/<name>.yaml`` is auto-discovered as scale ``<name>``; pass
``--scale-config <path>`` instead to use a file outside configs/
(run_pipeline.py forwards it to every stage).
"""

import os

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))

# Every key a scale file must define — load_scale fails fast with the exact
# missing keys so a hand-written experiment config can't half-apply.
REQUIRED_KEYS = {
    "data": ("num_cards",),
    "tokenize": (
        "seq_len",
        "target_eval_samples",
        "train_keep",
        "max_pretrain_windows",
        "holdout_keep",
        "shuffle_partitions",
    ),
    "model": ("d_model", "n_heads", "n_layers", "dim_ff"),
    "pretrain": ("epochs", "batch_size", "lr", "num_workers", "use_gpu", "use_fsdp"),
    "embed": ("num_workers", "use_gpu", "batch_size"),
    "downstream": ("num_workers", "use_gpu"),
}


def available_scales() -> list:
    return sorted(
        f[: -len(".yaml")] for f in os.listdir(CONFIG_DIR) if f.endswith(".yaml")
    )


def load_scale(name: str, config_path: str | None = None) -> dict:
    """Load and validate one scale's config (configs/<name>.yaml by default)."""
    import yaml

    path = config_path or os.path.join(CONFIG_DIR, f"{name}.yaml")
    if not os.path.isfile(path):
        raise SystemExit(
            f"no config for scale '{name}' ({path} not found) — available: "
            f"{', '.join(available_scales())}. Copy one of configs/*.yaml to "
            f"define a new scale, or pass --scale-config <path>."
        )
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for section, keys in REQUIRED_KEYS.items():
        missing = [k for k in keys if k not in (cfg.get(section) or {})]
        if missing:
            raise SystemExit(
                f"scale config {path} [{section}]: missing keys {missing} "
                f"(every file must be self-contained; see configs/mini.yaml)"
            )
    return cfg


def load_scales() -> dict:
    """name -> config for every discoverable scale, skipping invalid files.

    Lenient on purpose: a half-written experiment file in configs/ must not
    break importing src.paths (which derives SCALE_MAP from this).
    """
    scales = {}
    for name in available_scales():
        try:
            scales[name] = load_scale(name)
        except SystemExit:
            continue
    return scales


def add_scale_args(parser, default: str = "small") -> None:
    """The shared --scale/--scale-config wiring for stage scripts."""
    parser.add_argument("--scale", default=default)
    parser.add_argument(
        "--scale-config",
        default=None,
        help="explicit path to a scale YAML (default: configs/<scale>.yaml)",
    )
