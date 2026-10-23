"""CI test for the template: the no-GPU path must run end to end.

    python tests/test_template.py            # about 1 minute after setup_data.py

Deliberately does not touch GPUs or model weights. It checks that a clean
environment can download the scored results and reproduce the published
numbers, which is what every table and figure in the notebooks reads.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Share of poses within 2 A on late follow-ons, as published in the README.
EXPECTED = {
    ("sars2-mpro", "boltz2"): 91.7, ("sars2-mpro", "of3p2"): 87.7, ("sars2-mpro", "of3ray"): 62.1,
    ("mers-mpro", "boltz2"): 90.7, ("mers-mpro", "of3p2"): 73.5, ("mers-mpro", "of3ray"): 68.8,
}


def main():
    import notebook_helpers as wh

    failures = []

    run = wh.ours_run()
    if run is None:
        failures.append("no benchmarked fine-tune found; run scripts/setup_data.py first")

    for (target, model), want in sorted(EXPECTED.items()):
        try:
            late = wh.load_results(target).query("stage == 'late follow-on'")
            got = wh.success(late[f"{model}_lig_rmsd"])
        except Exception as e:                      # noqa: BLE001 - report, do not raise
            failures.append(f"{target}/{model}: {type(e).__name__}: {e}")
            continue
        if abs(got - want) > 0.1:
            failures.append(f"{target}/{model}: expected {want}%, got {got}%")
        else:
            print(f"ok  {target:12s} {model:10s} {got}%")

    curve = wh.loss_curve(run) if run else None
    if curve is None or len(curve) < 100:
        failures.append("loss curve missing or too short")
    else:
        print(f"ok  loss curve   {len(curve)} steps, loss {curve.loss_avg10.iloc[-1]:.3f}")

    for nb in ["01_ray_and_offline_inference.ipynb", "02_train_and_tune.ipynb", "03_serve.ipynb"]:
        if not (ROOT / nb).exists():
            failures.append(f"missing notebook {nb}")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
