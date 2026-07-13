"""Tests for R2 spec generation. Self-runnable: `python3 test_launcher.py`."""

import tempfile

import launcher
import registry as R


def _exp(**over):
    e = {"campaign": "08-esm2", "wave": 1, "rung": "proxy", "gpu": "A10G", "num_gpus": 1,
         "hours": 2, "entrypoint": "src/main.py", "base_config": "configs/proxy.yaml",
         "working_dir": "git://autoresearch@abc123", "image": "anyscale/ray:2.9-gpu",
         "flags": {"readout": "mlp"}, "seeds": [0, 1, 2]}
    e.update(over)
    return e


def test_run_name_is_greppable():
    n = launcher.run_name(_exp())
    assert n == "08-esm2-proxy-readoutmlp-s3", n
    print("ok  run name encodes campaign · rung · flags · seeds (Iron Rule #5)")


def test_money_saving_defaults_baked_in():
    spec = launcher.build_job_spec(_exp())
    w = spec["compute_config"]["worker_nodes"][0]
    assert w["min_nodes"] == 0 and w["market"] == "SPOT" and w["fallback_to_on_demand"]
    assert w["resources"]["CPU"] == 0                       # GPU fence
    assert w["instance_type"] == "g5.xlarge"                # A10G
    assert spec["compute_config"]["head_node"]["market"] == "ON_DEMAND"  # stateful head
    assert spec["timeout_s"] == 10800                       # 2h * 1.5 runtime kill
    print("ok  spot + scale-to-zero + CPU fence + on-demand head + timeout baked in")


def test_flags_default_off_only_set_emitted():
    spec = launcher.build_job_spec(_exp(flags={"readout": "mlp", "amp": True}))
    assert "--readout mlp" in spec["entrypoint"] and "--amp" in spec["entrypoint"]
    spec2 = launcher.build_job_spec(_exp(flags={}))
    assert "--readout" not in spec2["entrypoint"]           # default OFF -> absent
    print("ok  only set flags are emitted (one flag = one delta, default off)")


def test_over_cap_smoke_refused():
    with tempfile.TemporaryDirectory() as base:
        try:
            launcher.build_job_spec(_exp(rung="smoke", hours=3), base=base)  # 3 > 0.5 cap
            assert False, "should refuse"
        except launcher.BudgetError:
            pass
    print("ok  an over-cap smoke run is refused at build time")


def test_full_run_requires_pi_approval():
    with tempfile.TemporaryDirectory() as base:
        spec = launcher.build_job_spec(_exp(campaign="02-rl", wave=3, rung="full",
                                            gpu="H100", num_gpus=8, hours=15), base=base)
        assert spec["approval_required"] is True
        assert spec["estimate"]["a10g_equiv_hours"] == 660.0
        try:
            launcher.submit(spec)                            # no approval
            assert False, "should refuse without PI approval"
        except launcher.BudgetError:
            pass
        cli = launcher.submit(spec, pi_approved=True)        # with approval -> CLI, not run
        assert cli.startswith("anyscale job submit") and "02-rl-full" in cli
    print("ok  full run needs PI approval; submit returns CLI (never executes)")


def test_full_needs_approval_even_without_a_registry_base():
    """Self-audit regression: a full run built with no `base` (no spend context) must STILL be
    flagged approval-required — crossing into full is a policy fact, not a spend check. The
    earlier logic only set the flag when a base was passed, so submit() would have let a
    baseless full run through."""
    spec = launcher.build_job_spec(_exp(campaign="06", wave=3, rung="full",
                                        gpu="A100", num_gpus=8, hours=10))  # base=None
    assert spec["approval_required"] is True, spec
    try:
        launcher.submit(spec)
        assert False, "baseless full run must not submit without approval"
    except launcher.BudgetError:
        pass
    print("ok  full run flagged approval-required even with no registry base (audit fix)")


def test_flag_set_false_is_omitted():
    spec = launcher.build_job_spec(_exp(flags={"readout": "mlp", "amp": False, "pca": None}))
    assert "--amp" not in spec["entrypoint"] and "--pca" not in spec["entrypoint"]
    assert "amp" not in spec["name"] and "readoutmlp" in spec["name"]
    print("ok  a flag set False/None is omitted, not passed as '--flag False'")


def test_envelope_aware_refusal():
    with tempfile.TemporaryDirectory() as base:
        # spend 58 of a wave-1 (60) envelope, then a 5 A10G-eq proxy must refuse
        R.append_run(base, {"campaign": "01-hstu", "run_id": "s", "commit": "c", "rung": "proxy",
            "cost": {"gpu_hours": 58, "gpu_type": "A10G", "usd_est": 0, "spot": True},
            "eval_pin": "sha256:p", "seed_plan_commit": "s", "status": "SUCCEEDED"})
        try:
            launcher.build_job_spec(_exp(campaign="01-hstu", wave=1, rung="proxy", hours=5), base=base)
            assert False, "should refuse — over remaining envelope"
        except launcher.BudgetError:
            pass
    print("ok  build refuses a run that would exceed the remaining envelope")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} passed")
