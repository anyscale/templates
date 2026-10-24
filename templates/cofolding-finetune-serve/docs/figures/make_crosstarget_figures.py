"""Cross-target figures: what target-specific fine-tuning actually changed.

    python docs/figures/make_crosstarget_figures.py

Two figures, both drawn from the scored Parquet under $AIDDD_SHARED/results:

    docs/images/finetune_transfer.png   change in pose accuracy against the base
                                        model, per target and campaign stage
    docs/images/crosstarget_accuracy.png   absolute pose accuracy, every model

Accuracy is the share of curated binding events whose predicted ligand lands
within 2 A of the crystal pose, the same measure as scripts/compare_targets.py.

Each fine-tuned model is plotted against the checkpoint it started from, so the
bars read as "what fine-tuning on this data did", not as a ranking of models.
A model missing from the results is skipped, so the figures can be rebuilt
while a benchmark is still running.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GOOD, INK, INK2, MUTED, OUT, RULE, plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiddd_pipeline import SHARED, STAGES, TARGETS  # noqa: E402

# Every model, named by what it is and what it was fine-tuned on. The two
# fine-tuned entries name their training data because that is the variable
# under test: same architecture, different data.
MODELS = {
    "boltz2":    dict(name="Boltz-2\nnot fine-tuned",                       color="#1F5A96", base=None),
    "openbind0": dict(name="OpenBind-0 (OpenFold3)\nnot fine-tuned",        color="#E1A33B", base=None),
    "of3ft":     dict(name="OpenFold3 fine-tuned on\nEV-A71 2A fragments (published)",
                      color="#B3412E", base="openbind0",
                      # This checkpoint was trained from OF3-p2-145k, not from
                      # OpenBind-0, and OF3-p2 is not benchmarked here. So the bars
                      # are "what you get by deploying it instead of the best
                      # general model", not the effect of its own fine-tune.
                      note="started from OF3-p2, not from OpenBind-0:\n"
                           "read as deployment choice, not training effect"),
    "of3ray":    dict(name="OpenFold3 fine-tuned on\nEV-A71 2A fragments (ours, Ray Train)",
                      color="#7824D8", base="openbind0",
                      note="started from OpenBind-0: these bars are\nthe effect of the fine-tune itself"),
}
# Degradation keeps the repo's BAD red; improvement the GOOD green. Diverging
# pair with a neutral zero rule, never a hue at the midpoint.
WORSE = "#B3412E"

SHORT_TARGET = {"EV-A71 2A protease": "EV-A71 2A", "SARS-CoV-2 Mpro": "SARS-CoV-2 Mpro",
                "MERS-CoV Mpro": "MERS-CoV Mpro"}
SHORT_STAGE = {"fragment": "fragments", "early follow-on": "early follow-on",
               "late follow-on": "late follow-on"}


def load_scored(root: Path, tkey: str) -> pd.DataFrame:
    """Scored binding events on the curated set, from an explicit results root.

    Same filter as scripts/compare_targets.py. The files are read one by one
    because a Ray Data block whose column is entirely empty gets a null type.
    """
    files = sorted((root / tkey).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(root / tkey)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    meta = pd.read_csv(TARGETS[tkey].path("metadata"))
    keep = meta.pb_valid_ref.astype(bool) & ~meta.suspected_artefact.astype(bool)
    return df[df.complex_name.isin(set(meta.loc[keep, "complex_name"]))]


def accuracy_table(sources) -> pd.DataFrame:
    """One row per model, target and stage: n and the share of poses under 2 A.

    `sources` is a list of (results_root, {column_model_key: label}). Every fine-tuned
    checkpoint is benchmarked under the same `of3ray` column, so each run's results live
    in their own root and are relabelled here; otherwise three runs would collide into
    one column and silently overwrite each other.
    """
    rows = []
    for root, relabel in sources:
        for tkey, target in TARGETS.items():
            try:
                df = load_scored(root, tkey)
            except (FileNotFoundError, ValueError):
                continue
            for mkey, label in relabel.items():
                col = f"{mkey}_lig_rmsd"
                if col not in df or df[col].notna().sum() == 0:
                    continue
                for stage in STAGES:
                    s = df.loc[df.stage == stage, col].dropna()
                    if len(s) >= 10:      # a stage with a handful of events is noise
                        rows.append(dict(model=label, target=target.name, stage=stage,
                                         n=len(s), acc=100 * (s < 2).mean()))
    return pd.DataFrame(rows)


def register_runs(runs_arg: str) -> list:
    """Add one MODELS entry per fine-tuning run, so the figures treat them like any
    other model. Runs are compared against OpenBind-0, which they were trained from,
    and are drawn only in the transfer figure: the accuracy figure already carries four
    series, and a fifth hue would have to be invented rather than validated."""
    sources = []
    for pair in filter(None, (runs_arg or "").split(",")):
        label, _, root = pair.partition("=")
        label, root = label.strip(), root.strip()
        key = "run:" + label
        MODELS[key] = dict(name=label, color="#7824D8", base="openbind0",
                           note="started from OpenBind-0: these bars are\n"
                                "the effect of the fine-tune itself",
                           transfer_only=True)
        sources.append((Path(root), {"of3ray": key}))
    return sources


def ordered_cells(acc: pd.DataFrame):
    """Target/stage pairs in campaign order, only those every model was run on."""
    cells = []
    for tkey, target in TARGETS.items():
        for stage in STAGES:
            if ((acc.target == target.name) & (acc.stage == stage)).any():
                cells.append((target.name, stage))
    return cells


# ---------------------------------------------------------------------------
# Figure 1: what fine-tuning changed, against the checkpoint it started from
# ---------------------------------------------------------------------------
def figure_transfer(acc: pd.DataFrame, path: Path):
    tuned = [m for m in MODELS if MODELS[m]["base"] and (acc.model == m).any()]
    if not tuned:
        print("no fine-tuned model in the results yet, skipping", path.name)
        return
    cells = ordered_cells(acc)
    get = {(r.model, r.target, r.stage): r.acc for r in acc.itertuples()}

    width = 7.8 if len(tuned) == 1 else 5.8 * len(tuned)
    fig, axes = plt.subplots(1, len(tuned), figsize=(width, 4.8), sharey=True,
                             squeeze=False)
    axes = axes[0]
    y = np.arange(len(cells))[::-1]

    for ax, mkey in zip(axes, tuned):
        base = MODELS[mkey]["base"]
        deltas, labels = [], []
        for tname, stage in cells:
            a, b = get.get((mkey, tname, stage)), get.get((base, tname, stage))
            deltas.append(np.nan if a is None or b is None else a - b)
            labels.append(f"{SHORT_TARGET.get(tname, tname)}\n{SHORT_STAGE[stage]}")
        deltas = np.array(deltas, dtype=float)
        colors = [GOOD if d > 0 else WORSE for d in deltas]

        ax.barh(y, deltas, height=0.62, color=colors, zorder=3)
        ax.axvline(0, color=INK2, lw=0.9, zorder=4)
        for yi, d in zip(y, deltas):
            if np.isnan(d):
                continue
            off = 1.6 if d > 0 else -1.6
            ax.text(d + off, yi, f"{d:+.1f}", va="center", ha="left" if d > 0 else "right",
                    fontsize=9.5, color=INK, zorder=5)

        ax.set_yticks(y, labels, fontsize=9)
        ax.set_xlabel(f"change in poses under 2 A vs {MODELS[base]['name'].split(chr(10))[0]}"
                      "\n(percentage points)", fontsize=9)
        ax.set_title(MODELS[mkey]["name"].replace("\n", " "), color=INK, pad=22)
        # Which checkpoint this model actually started from decides whether the
        # bars measure fine-tuning or just a deployment choice. Say so on the panel.
        ax.text(0, 1.015, MODELS[mkey]["note"], transform=ax.transAxes, fontsize=8,
                color=MUTED, va="bottom", ha="left", linespacing=1.3)
        lim = max(60, np.nanmax(np.abs(deltas)) * 1.35)
        ax.set_xlim(-lim, lim)
        ax.grid(axis="x", color=RULE, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)

    # One shared reading of the chart, in ink rather than in the bar colours.
    fig.text(0.012, 0.975, "Fine-tuning on one target's fragments:\n"
             "a large gain there, losses everywhere else",
             fontsize=11.5, fontweight="bold", color=INK, va="top", ha="left", linespacing=1.35)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# Figure 2: absolute accuracy, every model
# ---------------------------------------------------------------------------
def figure_accuracy(acc: pd.DataFrame, path: Path):
    cells = ordered_cells(acc)
    present = [m for m in MODELS
               if (acc.model == m).any() and not MODELS[m].get("transfer_only")]
    get = {(r.model, r.target, r.stage): r.acc for r in acc.itertuples()}

    fig, ax = plt.subplots(figsize=(max(9.0, 1.35 * len(cells)), 4.8))
    x = np.arange(len(cells), dtype=float)
    w = 0.80 / len(present)

    for i, mkey in enumerate(present):
        vals = [get.get((mkey, t, s), np.nan) for t, s in cells]
        off = (i - (len(present) - 1) / 2) * w
        # 2px surface gap between adjacent bars: drawn as a slightly narrow bar.
        ax.bar(x + off, vals, width=w * 0.88, color=MODELS[mkey]["color"],
               label=MODELS[mkey]["name"].replace("\n", " "), zorder=3)

    # Direct-label only the fragment cells, where the whole story sits.
    for i, mkey in enumerate(present):
        off = (i - (len(present) - 1) / 2) * w
        for xi, (t, s) in zip(x, cells):
            if s != "fragment":
                continue
            v = get.get((mkey, t, s))
            if v is not None:
                ax.text(xi + off, v + 1.6, f"{v:.0f}", ha="center", va="bottom",
                        fontsize=8.5, color=INK, zorder=5)

    ax.set_xticks(x, [f"{SHORT_TARGET.get(t, t)}\n{SHORT_STAGE[s]}" for t, s in cells],
                  fontsize=9)
    ax.set_ylabel("binding events with the ligand under 2 A (%)", fontsize=9.5)
    ax.set_ylim(0, 104)
    ax.set_xlim(-0.62, len(cells) - 0.38)
    ax.grid(axis="y", color=RULE, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)

    # Title above, legend below it: anchoring both to the axes collides them.
    fig.text(0.012, 0.975, "Pose accuracy by target and campaign stage",
             fontsize=11.5, fontweight="bold", color=INK, va="top", ha="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.13), ncol=2, fontsize=8.5,
              handlelength=1.1, columnspacing=1.8, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(SHARED / "results"),
                    help="root holding one <target>/ directory of scored Parquet per target, "
                         "for the models that are not fine-tuned here")
    ap.add_argument("--runs", default=None,
                    help="fine-tuning runs to add, as label=results_root pairs separated by "
                         "commas, e.g. 'A 66 frags=/mnt/.../results_ft-a5-ev71-frags-66'. "
                         "Each is read from its own root and plotted under its own label")
    args = ap.parse_args()

    sources = [(Path(args.results), {m: m for m in list(MODELS)})]
    sources += register_runs(args.runs)
    acc = accuracy_table(sources)
    if acc.empty:
        raise SystemExit("No scored results found. Run aiddd_pipeline.py first.")
    print(acc.pivot_table(index=["target", "stage"], columns="model", values="acc")
             .round(1).to_string())
    OUT.mkdir(parents=True, exist_ok=True)
    figure_transfer(acc, OUT / "finetune_transfer.png")
    figure_accuracy(acc, OUT / "crosstarget_accuracy.png")


if __name__ == "__main__":
    main()
