"""Report whether shared storage holds everything the notebooks read.

    python scripts/check_data.py            # the CPU path every attendee runs
    python scripts/check_data.py --gpus     # also the weights USE_GPUS = True needs

Checks the data, not the code: run it on a new cloud before the room arrives.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", action="store_true", help="also check model weights and the fine-tune")
    args = ap.parse_args()

    from aiddd_pipeline import (SHARED, TARGETS, BOLTZ_CACHE, OF3_OB_CKPT, OF3_P2_CKPT,
                                OF3_FT_CKPT, OF3_OURS_CKPT)
    import notebook_helpers as wh

    print(f"Shared data directory: {SHARED}\n")
    bad = []

    def check(ok, label, detail=""):
        print(f"  {'ok     ' if ok else 'MISSING'} {label}{'  ' + detail if detail else ''}")
        if not ok:
            bad.append(label)

    print("notebook 01 setup gate (USE_GPUS = False)")
    for t in wh.TARGET_KEYS:
        p = TARGETS[t].path("metadata")
        check(p.exists(), f"{t} metadata")

    print("\nscored results, the source of every table and figure")
    run = wh.ours_run()
    check(run is not None, "benchmarked fine-tune run", str(run or ""))
    for t in wh.TARGET_KEYS:
        try:
            n = len(wh.load_results(t))
            check(n > 0, f"{t} results", f"{n} rows")
        except Exception as e:                                  # noqa: BLE001
            check(False, f"{t} results", f"{type(e).__name__}: {e}")
    curve = wh.loss_curve(run) if run else None
    check(curve is not None and len(curve) >= 100, "fine-tune loss curve",
          f"{len(curve)} steps" if curve is not None else "")

    print("\ncached poses, the 3D viewers (absent: viewers show the crystal only)")
    for t in wh.TARGET_KEYS:
        try:
            df = wh.load_results(t)
        except Exception:
            continue
        for col in [c for c in df.columns if c.endswith("_cif")]:
            s = df[col].dropna()
            n = sum(1 for v in s if Path(str(v)).exists())
            print(f"  {'ok     ' if n else 'none   '} {t} {col}: {n}/{len(s)}")

    if args.gpus:
        print("\nmodel weights (USE_GPUS = True only)")
        for p, label in [(BOLTZ_CACHE, "Boltz-2 cache"), (OF3_OB_CKPT, "OpenBind-0"),
                         (OF3_P2_CKPT, "OpenFold3 145k"), (OF3_FT_CKPT, "OpenBind fine-tune"),
                         (OF3_OURS_CKPT, "our fine-tune")]:
            check(Path(p).exists(), label, str(p))

    print()
    if bad:
        print(f"INCOMPLETE: {len(bad)} missing -> run `python scripts/setup_data.py`"
              + ("" if args.gpus else " (add --gpus here to check weights)"))
        return 1
    print("complete: every notebook has what it reads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
