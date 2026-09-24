#!/usr/bin/env python3
"""The numbers that live in several files, checked against each other.

Four decisions in this template are written down in more than one place, because
each place is read by a different program. Nothing else notices when they drift,
and each drift fails the same quiet way: a task that pends forever, or a table
scored over a region the reads do not cover.

* **The node ceiling.** pipeline/conf/base.config (``resourceLimits``, which
  Nextflow clamps to), pipeline/conf/ray.config (``maxNode*``, which the executor
  rejects over) and configs/nextflow-genomics-on-ray/{aws,gce}.yaml (the instance
  types). The ceiling has to be what the largest worker can *schedule*: Ray
  2.58.0 advertises about 70% of a node's free memory as ``memory``
  (DEFAULT_OBJECT_STORE_MEMORY_PROPORTION = 0.3), so the check allows 0.7 x the
  instance's memory, less 5% for the OS and the container.
* **The GPU label's ceiling**, against the GPU worker, since a GPU task can land
  nowhere else.
* **The scale regions** in main.nf's ``scales()`` and tools/stage-demo-data.sh.
* **The plugin version** in ray.config, nf-ray-plugin/build.gradle and
  nf_ray/_version.py.

Instance sizes are from the providers' published specs (AWS EC2 instance types;
GCP machine families), in GiB, and are the one thing here that is typed in by
hand. Plain script, like the others in this directory.
"""

from __future__ import annotations

import os
import re
import sys
import traceback

# The template directory: CWD under tests.sh, where rayapp (so CI) flattens
# templates/<name>/ and tests/<name>/ together; the repo layout otherwise.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_TEMPLATE = next(
    (
        d
        for d in (os.getcwd(), _HERE, os.path.join(_REPO, "templates", "nextflow-genomics-on-ray"))
        if os.path.isdir(os.path.join(d, "pipeline"))
    ),
    os.path.join(_REPO, "templates", "nextflow-genomics-on-ray"),
)
#: Compute configs exist only in a repo checkout; rayapp does not ship them to the
#: test cluster. The checks that need them say so and skip there.
_CONFIGS = os.path.join(_REPO, "configs", "nextflow-genomics-on-ray")
_HAVE_CONFIGS = os.path.isfile(os.path.join(_CONFIGS, "aws.yaml"))

#: vCPU, memory GiB, GPUs, accelerator.
INSTANCES = {
    "m5.2xlarge": (8, 32, 0, None),
    "r6i.4xlarge": (16, 128, 0, None),
    "g6.2xlarge": (8, 32, 1, "L4"),
    "n2-standard-8": (8, 32, 0, None),
    "n2-highmem-16": (16, 128, 0, None),
    "g2-standard-8-nvidia-l4-1": (8, 32, 1, "L4"),
}

#: Ray's share of free memory for tasks, and an allowance for what is not free.
RAY_MEMORY_SHARE = 0.7
OS_ALLOWANCE = 0.95

_FAILURES: list[str] = []


def check(name: str, needs_configs: bool = False):
    def wrap(fn):
        if needs_configs and not _HAVE_CONFIGS:
            print(f"skip {name} (no {_CONFIGS} here)")
            return fn
        try:
            fn()
        except Exception:
            _FAILURES.append(name)
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
        return fn

    return wrap


def read(*parts: str) -> str:
    with open(os.path.join(*parts)) as handle:
        return handle.read()


def schedulable_gib(memory_gib: float) -> float:
    return memory_gib * RAY_MEMORY_SHARE * OS_ALLOWANCE


def worker_groups(cloud_file: str) -> list[dict]:
    """The worker_nodes of a compute config, without needing PyYAML."""
    groups, current = [], None
    in_workers = False
    for line in read(_CONFIGS, cloud_file).splitlines():
        if line.startswith("worker_nodes:"):
            in_workers = True
            continue
        if in_workers and line and not line.startswith((" ", "-", "#")):
            in_workers = False
        if not in_workers:
            continue
        stripped = line.strip()
        if stripped.startswith("- name:"):
            current = {"name": stripped.split(":", 1)[1].strip()}
            groups.append(current)
        elif current is not None and ":" in stripped and not stripped.startswith("#"):
            key, value = (p.strip() for p in stripped.split(":", 1))
            current[key] = value
    return groups


def label_block(text: str, label: str) -> str:
    """The body of ``withLabel:<label> { ... }``, braces matched: the closures
    inside (``{ 6 * task.attempt }``) have braces of their own."""
    start = text.index("{", text.index(f"withLabel:{label}"))
    depth = 0
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start + 1 : i]
    raise AssertionError(f"unbalanced braces after withLabel:{label}")


def base_limits(block: str = "") -> dict[str, float]:
    text = read(_TEMPLATE, "pipeline", "conf", "base.config")
    # A label's own, or else the process-wide one (everything before the labels).
    text = label_block(text, block) if block else text[: text.index("withLabel:")]
    match = re.search(r"resourceLimits\s*=\s*\[\s*cpus:\s*(\d+),\s*memory:\s*(\d+)\.GB", text)
    assert match, f"no resourceLimits found{' in ' + block if block else ''}"
    return {"cpus": float(match.group(1)), "memory_gib": float(match.group(2))}


def ray_ceiling() -> dict[str, float]:
    text = read(_TEMPLATE, "pipeline", "conf", "ray.config")
    out = {}
    for key, name in (("maxNodeCpus", "cpus"), ("maxNodeMemoryGb", "memory_gib"),
                      ("maxNodeGpus", "gpus")):
        match = re.search(rf"^\s*{key}\s*=\s*(\d+)", text, re.M)
        assert match, f"ray.config has no {key}"
        out[name] = float(match.group(1))
    return out


@check("ceiling: base.config's resourceLimits and ray.config's maxNode* are one number")
def _() -> None:
    limits, ceiling = base_limits(), ray_ceiling()
    assert limits["cpus"] == ceiling["cpus"], (limits, ceiling)
    assert limits["memory_gib"] == ceiling["memory_gib"], (limits, ceiling)


@check("ceiling: the largest CPU worker on each cloud can schedule it", needs_configs=True)
def _() -> None:
    ceiling = ray_ceiling()
    for cloud in ("aws.yaml", "gce.yaml"):
        cpu_groups = [g for g in worker_groups(cloud) if INSTANCES[g["instance_type"]][2] == 0]
        assert cpu_groups, f"{cloud}: no CPU worker group"
        best = max(cpu_groups, key=lambda g: INSTANCES[g["instance_type"]][1])
        vcpu, mem, _gpus, _acc = INSTANCES[best["instance_type"]]
        assert vcpu >= ceiling["cpus"], f"{cloud}: {best['instance_type']} has {vcpu} vCPU"
        room = schedulable_gib(mem)
        assert ceiling["memory_gib"] <= room, (
            f"{cloud}: {best['instance_type']} schedules about {room:.0f} GiB, "
            f"but the ceiling is {ceiling['memory_gib']:.0f}"
        )


@check("ceiling: a 64 GiB worker would not have been enough (the bug this replaces)")
def _() -> None:
    # Pins the arithmetic, not a config: 72 GB process_high against an m5.4xlarge.
    assert schedulable_gib(64) < 72


@check("gpu: process_gpu's own ceiling fits the GPU worker on each cloud", needs_configs=True)
def _() -> None:
    limits, ceiling = base_limits("process_gpu"), ray_ceiling()
    for cloud in ("aws.yaml", "gce.yaml"):
        gpu_groups = [g for g in worker_groups(cloud) if INSTANCES[g["instance_type"]][2] > 0]
        assert len(gpu_groups) == 1, f"{cloud}: expected one GPU group"
        vcpu, mem, gpus, acc = INSTANCES[gpu_groups[0]["instance_type"]]
        assert limits["cpus"] <= vcpu, (cloud, limits, vcpu)
        assert limits["memory_gib"] <= schedulable_gib(mem), (cloud, limits, mem)
        assert gpus == ceiling["gpus"], (cloud, gpus, ceiling)
        assert acc == "L4", f"{cloud}: the pipeline asks for nvidia-l4, this group has {acc}"
    base = read(_TEMPLATE, "pipeline", "conf", "base.config")
    assert "type: 'nvidia-l4'" in base
    assert "defaultAccelerator = 'nvidia-l4'" in read(_TEMPLATE, "pipeline", "conf", "ray.config")


@check("labels: every nf-core label but process_high_memory fits at attempt 1")
def _() -> None:
    # process_high_memory is left at nf-core's 200 GB on purpose and clamped;
    # everything else should run as asked on its first attempt.
    base = read(_TEMPLATE, "pipeline", "conf", "base.config")
    limits = base_limits()
    for label in ("process_single", "process_low", "process_medium", "process_high"):
        block = label_block(base, label)
        cpus = re.search(r"cpus\s*=\s*\{?\s*(\d+)", block)
        mem = re.search(r"memory\s*=\s*\{\s*(\d+)\s*\.GB", block)
        assert cpus and mem, label
        assert float(cpus.group(1)) <= limits["cpus"], (label, cpus.group(1))
        assert float(mem.group(1)) <= limits["memory_gib"], (label, mem.group(1))


@check("configs: the head is unschedulable and the CPU group starts warm", needs_configs=True)
def _() -> None:
    for cloud in ("aws.yaml", "gce.yaml"):
        text = read(_CONFIGS, cloud)
        head = text[text.index("head_node:") : text.index("worker_nodes:")]
        assert re.search(r"CPU:\s*0\b", head), f"{cloud}: head needs resources: {{CPU: 0}}"
        groups = {g["name"]: g for g in worker_groups(cloud)}
        assert groups["cpu-worker"].get("min_nodes") == "1", cloud
        assert groups["gpu-worker"].get("min_nodes", "0") == "0", cloud


@check("scales: main.nf and stage-demo-data.sh agree on every region")
def _() -> None:
    main = read(_TEMPLATE, "pipeline", "main.nf")
    stage = read(_TEMPLATE, "tools", "stage-demo-data.sh")
    scale_re = r"^\s*(quick|standard|full)\s*:\s*\[\s*region:\s*'([^']+)'"
    in_main = dict(re.findall(scale_re, main, re.M))
    region_re = r'^REGION_(QUICK|STANDARD|FULL)="\$\{REGION_\1:-([^}]+)\}"'
    in_stage = {m.group(1).lower(): m.group(2) for m in re.finditer(region_re, stage, re.M)}
    assert set(in_main) == {"quick", "standard", "full"}, in_main
    assert in_main == in_stage, f"main.nf {in_main} != stage-demo-data.sh {in_stage}"


@check("plugin: one version in ray.config, build.gradle and nf_ray")
def _() -> None:
    ray_cfg = re.search(r"id\s+'nf-ray@([^']+)'", read(_TEMPLATE, "pipeline", "conf", "ray.config"))
    build_gradle = read(_TEMPLATE, "nf-ray-plugin", "build.gradle")
    gradle = re.search(r"^version\s*=\s*'([^']+)'", build_gradle, re.M)
    python = re.search(r'__version__\s*=\s*"([^"]+)"', read(_TEMPLATE, "nf_ray", "_version.py"))
    assert ray_cfg and gradle and python
    assert ray_cfg.group(1) == gradle.group(1) == python.group(1), (
        ray_cfg.group(1), gradle.group(1), python.group(1)
    )


if _FAILURES:
    print(f"\n{len(_FAILURES)} failing: {', '.join(_FAILURES)}", file=sys.stderr)
    sys.exit(1)
print("\nall config agreement checks passed")
