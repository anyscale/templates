#!/usr/bin/env python3
"""Offline unit tests for the Ray side of the Nextflow executor: no cluster, no Ray."""

from __future__ import annotations

import os
import re
import socket
import sys
import tempfile
import traceback

# rayapp flattens templates/<name>/ and tests/<name>/ into the CWD; fall back to the repo layout.
_HERE = os.path.dirname(os.path.abspath(__file__))
_TEMPLATE = os.path.abspath(
    os.path.join(_HERE, "..", "..", "templates", "nextflow-genomics-on-ray")
)
for candidate in (os.getcwd(), _HERE, _TEMPLATE):
    if os.path.isdir(os.path.join(candidate, "nf_ray")):
        sys.path.insert(0, candidate)
        _TEMPLATE = candidate
        break

from nf_ray import directives, envs, errors, resources  # noqa: E402
from nf_ray.config import (  # noqa: E402
    Config,
    default_socket_path,
    is_shared_storage,
    shared_storage_warning,
)

GIB = 1 << 30
MIB = 1 << 20

_FAILURES: list[str] = []


def check(name: str):
    def wrap(fn):
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


# As RayExecutor.groovy emits it: one directive per line, then a body with a decoy header.
REAL_HEADER = """\
#!/bin/bash
#RAY -name nf-GATK4_HAPLOTYPECALLER_HG002_chr20_3
#RAY -cpus 6
#RAY -memory 36864
#RAY -time 480
#RAY -accelerator nvidia-l4
#RAY -gpus 1
#RAY -resources {"nvme":1}
NXF_ENTRY=${1:-nxf_main}
nxf_main() {
    cat <<'EOF'
#RAY -cpus 999
EOF
}
"""


@check("header: parses every directive the plugin emits")
def _() -> None:
    d = directives.parse_header(REAL_HEADER)
    assert d.name == "nf-GATK4_HAPLOTYPECALLER_HG002_chr20_3", d.name
    assert d.cpus == 6.0, d.cpus
    assert d.memory_mb == 36864, d.memory_mb
    assert d.gpus == 1.0, d.gpus
    assert d.accelerator == "nvidia-l4", d.accelerator
    assert d.resources == '{"nvme":1}', d.resources
    assert d.time_minutes == 480, d.time_minutes


@check("header: stops at the script body")
def _() -> None:
    # The decoy must be parseable (column 0, in a heredoc), or this passes without the cutoff.
    assert "\n#RAY -cpus 999\n" in REAL_HEADER, "fixture must contain a parseable decoy"
    assert directives.parse_header(REAL_HEADER).cpus == 6.0


@check("header: absent directives fall back to WDL-free defaults")
def _() -> None:
    d = directives.parse_header("#!/bin/bash\n#RAY -name solo\n")
    assert d.cpus == 1.0 and d.memory_mb == 0 and d.gpus == 0.0
    assert d.accelerator == "" and d.image == ""


@check("header: unknown directives are kept, not dropped")
def _() -> None:
    d = directives.parse_header("#!/bin/bash\n#RAY -cpus 2\n#RAY -somethingnew yes\n")
    assert d.extra == {"somethingnew": "yes"}, d.extra


@check("header: a non-numeric value is an error, not a silent 1 cpu")
def _() -> None:
    try:
        directives.parse_header("#!/bin/bash\n#RAY -cpus lots\n")
    except directives.DirectiveError as exn:
        assert "lots" in str(exn)
    else:
        raise AssertionError("expected DirectiveError")


@check("header: reads from a real file, only the head of it")
def _() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, ".command.run")
        with open(path, "w") as fh:
            fh.write(REAL_HEADER)
            fh.write("\n# padding\n" * 5000)
        assert directives.parse_script(path).cpus == 6.0


@check("header: the scan is bounded even with no body line to stop at")
def _() -> None:
    # An all-comment file never trips the body check, so only the line limit bounds the scan.
    limit = directives._MAX_HEADER_LINES
    comments = ["# filler"] * (limit + 50)
    comments.append("#RAY -cpus 64")  # past the limit: must not be read
    d = directives.parse_header("#!/bin/bash\n#RAY -cpus 2\n" + "\n".join(comments))
    assert d.cpus == 2.0, d.cpus

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, ".command.run")
        with open(path, "w") as fh:
            fh.write("#!/bin/bash\n#RAY -cpus 2\n" + "\n".join(comments))
        assert directives.parse_script(path).cpus == 2.0


@check("resources: cpus, memory and gpus map onto Ray options")
def _() -> None:
    req = resources.build_request(directives.parse_header(REAL_HEADER))
    opts = req.options()
    assert opts["num_cpus"] == 6.0
    assert opts["memory"] == 36864 * MIB
    assert opts["num_gpus"] == 1.0
    assert opts["accelerator_type"] == "L4", opts
    assert opts["resources"] == {"nvme": 1.0}


@check("resources: nvidia.com/gpu means 'any GPU', not a model named that")
def _() -> None:
    # The Kubernetes spelling; passed through as accelerator_type, it matches no node.
    d = directives.parse_header("#RAY -gpus 1\n#RAY -accelerator nvidia.com/gpu\n")
    assert resources.build_request(d).accelerator_type is None

    withdefault = resources.build_request(d, default_accelerator_type="nvidia-l4")
    assert withdefault.accelerator_type == "L4"


@check("resources: Ray's own accelerator spellings pass through")
def _() -> None:
    d = directives.parse_header("#RAY -gpus 2\n#RAY -accelerator A10G\n")
    assert resources.build_request(d).accelerator_type == "A10G"


@check("resources: an accelerator with no GPU request is dropped")
def _() -> None:
    d = directives.parse_header("#RAY -cpus 1\n#RAY -accelerator nvidia-l4\n")
    assert resources.build_request(d).accelerator_type is None


@check("resources: time is parsed, reported, and not enforced")
def _() -> None:
    req = resources.build_request(directives.parse_header(REAL_HEADER))
    assert req.time_minutes == 480
    assert "time" not in " ".join(req.options())
    assert req.describe()["time_minutes_ignored"] == 480


@check("resources: malformed ext.ray_resources fails loudly")
def _() -> None:
    d = directives.parse_header("#RAY -cpus 1\n#RAY -resources notjson\n")
    try:
        resources.build_request(d)
    except ValueError as exn:
        assert "ray_resources" in str(exn)
    else:
        raise AssertionError("expected ValueError")


# nf-core's process_high_memory request, which pends forever on a 64 GiB worker.
NFCORE_HIGH_MEMORY = directives.parse_header(
    "#RAY -name nf-GATK4_MARKDUPLICATES\n#RAY -cpus 12\n#RAY -memory 204800\n"
)


@check("schedulable: an oversized request is rejected, not left to pend")
def _() -> None:
    limits = resources.NodeLimits(
        cpus=16, memory_bytes=64 * GIB, gpus=1, source="test"
    )
    req = resources.build_request(NFCORE_HIGH_MEMORY)
    try:
        resources.check_schedulable(req, limits, process="GATK4_MARKDUPLICATES")
    except resources.Unschedulable as exn:
        text = str(exn)
        # Ray says nothing about a request that never fits, so the message must.
        assert "200.0 GiB" in text, text
        assert "64.0 GiB" in text, text
        assert "aws,gce" in text, text
        assert "base.config" in text, text
        assert "NF_RAY_CLAMP_RESOURCES" in text, text
    else:
        raise AssertionError("expected Unschedulable")


@check("schedulable: a request that fits is returned untouched")
def _() -> None:
    limits = resources.NodeLimits(cpus=16, memory_bytes=64 * GIB, gpus=1, source="test")
    req = resources.build_request(directives.parse_header("#RAY -cpus 6\n#RAY -memory 36864\n"))
    assert resources.check_schedulable(req, limits) is req


@check("schedulable: clamping shrinks to the node instead of raising")
def _() -> None:
    limits = resources.NodeLimits(cpus=16, memory_bytes=64 * GIB, gpus=0, source="test")
    req = resources.build_request(NFCORE_HIGH_MEMORY)
    clamped = resources.check_schedulable(req, limits, clamp=True)
    assert clamped.num_cpus == 12.0
    assert clamped.memory == 64 * GIB


@check("schedulable: an unknown ceiling submits rather than guessing")
def _() -> None:
    # A cold cluster has no workers yet; refusing would break every cold start.
    req = resources.build_request(NFCORE_HIGH_MEMORY)
    assert resources.check_schedulable(req, resources.NodeLimits()) is req


class _FakeOOM(Exception):
    pass


_FakeOOM.__name__ = "OutOfMemoryError"


class _FakeNodeDied(Exception):
    pass


_FakeNodeDied.__name__ = "NodeDiedError"


class _FakeSubclass(_FakeNodeDied):
    pass


_FakeSubclass.__name__ = "SomeFutureRayError"


@check("errors: Ray's memory monitor maps onto nf-core's OOM retry code")
def _() -> None:
    assert errors.exit_code_for(_FakeOOM()) == 137
    assert errors.is_retryable(_FakeOOM())


@check("errors: preemption is infrastructure, and retryable")
def _() -> None:
    assert errors.exit_code_for(_FakeNodeDied()) == errors.EXIT_NODE_LOST
    assert errors.EXIT_NODE_LOST in errors.NFCORE_RETRYABLE


@check("errors: an unrecognized subclass inherits its parent's code")
def _() -> None:
    assert errors.exit_code_for(_FakeSubclass()) == errors.EXIT_NODE_LOST


@check("errors: an unknown framework failure is NOT retried")
def _() -> None:
    assert errors.exit_code_for(RuntimeError("who knows")) == errors.EXIT_FRAMEWORK
    assert not errors.is_retryable(RuntimeError("who knows"))


@check("errors: every mapped code lands on the intended side of nf-core's band")
def _() -> None:
    # nf-core's conf/base.config predicate, verbatim.
    def nfcore_retries(status: int) -> bool:
        return status in set(range(130, 146)) | {104} | set(range(175, 178))

    for name, code in errors.RAY_ERROR_EXIT_CODES.items():
        assert nfcore_retries(code), f"{name} -> {code} would not be retried"
    assert not nfcore_retries(errors.EXIT_FRAMEWORK)
    for code in range(100, 200):
        assert nfcore_retries(code) == (code in errors.NFCORE_RETRYABLE), code


@check("errors: the log line names the code and whether it retries")
def _() -> None:
    text = errors.describe(_FakeOOM("worker killed"))
    assert "OutOfMemoryError" in text and "137" in text and "retryable" in text


@check("image: an unmapped ext.image fails rather than silently substituting")
def _() -> None:
    config = Config(image_map={}, image_fallback="error")
    try:
        envs.resolve_image("quay.io/biocontainers/fastp:0.23.4--hadf994f_2", config)
    except envs.ImageUnavailable as exn:
        text = str(exn)
        assert "NF_RAY_IMAGE_MAP" in text
        assert "probe-image" in text
    else:
        raise AssertionError("expected ImageUnavailable")


@check("image: a mapping and a '*' catch-all both resolve")
def _() -> None:
    assert envs.resolve_image("declared", Config(image_map={"declared": "rebuilt"})) == "rebuilt"
    assert envs.resolve_image("anything", Config(image_map={"*": "house"})) == "house"


@check("image: fallback=ignore runs in the cluster's own image")
def _() -> None:
    assert envs.resolve_image("declared", Config(image_fallback="ignore")) == ""


@check("image: no ext.image means no runtime_env at all")
def _() -> None:
    assert envs.build_runtime_env(directives.parse_header("#RAY -cpus 1\n"), Config()) == {}


@check("config: NF_RAY_* env vars are read with their documented types")
def _() -> None:
    saved = dict(os.environ)
    try:
        os.environ.update(
            {
                "NF_RAY_WORK_DIR": "/mnt/cluster_storage/nf-work",
                "NF_RAY_CLAMP_RESOURCES": "true",
                "NF_RAY_SCHEDULING_STRATEGY": "spread",
                "NF_RAY_MAX_NODE_MEMORY_GB": "64",
                "NF_RAY_IMAGE_MAP": '{"a":"b"}',
                "NF_RAY_EXTRA_RESOURCES": '{"nvme":1}',
            }
        )
        config = Config.from_env()
        assert config.clamp_resources is True
        assert config.scheduling_strategy == "SPREAD"
        assert config.max_node_memory_gb == 64.0
        assert config.image_map == {"a": "b"}
        assert config.extra_resources == {"nvme": 1.0}
        assert config.socket_path == default_socket_path("/mnt/cluster_storage/nf-work")
    finally:
        os.environ.clear()
        os.environ.update(saved)


@check("config: the daemon socket is node-local, per work dir, and short")
def _() -> None:
    # Not on NFS (bind and flock are unreliable there), and sun_path is 108 bytes (104 on macOS).
    a = default_socket_path("/mnt/cluster_storage/nf-work")
    b = default_socket_path("/mnt/cluster_storage/other-run")
    assert a.startswith("/tmp/nf-ray-") and a.endswith(".sock"), a
    assert a != b, "two work dirs must not share a daemon"
    assert a == default_socket_path("/mnt/cluster_storage/nf-work/"), "must not depend on a slash"

    deep = "/mnt/cluster_storage/" + "/".join(["a-rather-long-directory-name"] * 10)
    path = default_socket_path(deep)
    assert len(path) == len(a) < 104, path
    old_style = os.path.join(deep, ".nf-ray.sock")
    for candidate, should_bind in ((path, True), (old_style, False)):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(candidate)
            except OSError as exn:
                assert not should_bind, f"{candidate}: {exn}"
                # Refused for its length, before the missing directory matters.
                assert "too long" in str(exn), exn
            else:
                assert should_bind, f"{candidate} bound despite being {len(candidate)} bytes"
                os.unlink(candidate)


@check("config: Ray-level retries are off by default, from both directions")
def _() -> None:
    # A hidden Ray retry would rerun an OOM at the same size, defeating task.attempt scaling.
    assert Config().task_max_retries == 0
    saved = dict(os.environ)
    try:
        os.environ.pop("NF_RAY_TASK_MAX_RETRIES", None)
        assert Config.from_env().task_max_retries == 0
        os.environ["NF_RAY_TASK_MAX_RETRIES"] = "2"
        assert Config.from_env().task_max_retries == 2, "must still be overridable"
    finally:
        os.environ.clear()
        os.environ.update(saved)


@check("config: a bad enum value fails at startup, not at task 300")
def _() -> None:
    saved = dict(os.environ)
    try:
        os.environ["NF_RAY_IMAGE_FALLBACK"] = "maybe"
        try:
            Config.from_env()
        except ValueError as exn:
            assert "IMAGE_FALLBACK" in str(exn)
        else:
            raise AssertionError("expected ValueError")
    finally:
        os.environ.clear()
        os.environ.update(saved)


@check("storage: the shared-storage check is not gated on live node count")
def _() -> None:
    assert is_shared_storage("/mnt/cluster_storage/nf-work")
    assert is_shared_storage("/mnt/user_storage")
    assert not is_shared_storage("/mnt/local_storage/nf-work")
    assert not is_shared_storage("/home/ray/work")
    assert not is_shared_storage("/mnt/cluster_storage_backup/x")

    warning = shared_storage_warning("/mnt/local_storage/nf-work")
    assert "second node" in warning, warning
    assert "workDir = '/mnt/cluster_storage/nf-work'" in warning
    assert shared_storage_warning("/mnt/cluster_storage/nf-work") == ""


def expected_smoke_checksums(shards: int) -> list[int]:
    """What pipeline/smoke.nf must produce, derived in closed form rather than recorded."""
    return [1_000_000 * i + 500_500 for i in range(shards)]


@check("smoke: the cross-dispatch oracle has a closed form")
def _() -> None:
    assert expected_smoke_checksums(4) == [500_500, 1_500_500, 2_500_500, 3_500_500]
    for i in range(4):
        assert expected_smoke_checksums(4)[i] == sum(range(i * 1000 + 1, i * 1000 + 1001))


@check("smoke: smoke.nf still generates the range the oracle assumes")
def _() -> None:
    smoke = os.path.join(_TEMPLATE, "pipeline", "smoke.nf")
    if not os.path.exists(smoke):
        raise AssertionError(f"{smoke} is missing")
    with open(smoke) as handle:
        text = handle.read()
    assert "${idx} * 1000 + 1" in text, "smoke.nf shard range changed"
    assert "${idx} * 1000 + 1000" in text, "smoke.nf shard range changed"
    assert "export LC_ALL=C" in text, "smoke.nf must pin locale or the oracle drifts"


@check("smoke: the gather runs a bin/ script, so the smoke run checks bin/ reaches workers")
def _() -> None:
    # Without the executor's copy of bin/, COLLECT_BENCHMARK fails on a worker: command not found.
    with open(os.path.join(_TEMPLATE, "pipeline", "smoke.nf")) as handle:
        text = handle.read()
    collect = text[text.index("process COLLECT") :]
    assert "smoke_collect.sh ${reports}" in collect, "smoke.nf COLLECT no longer calls bin/"
    script = os.path.join(_TEMPLATE, "pipeline", "bin", "smoke_collect.sh")
    assert os.path.isfile(script), f"{script} is missing"


@check("plugin: tasks run bin/ from an executable copy on shared storage")
def _() -> None:
    # By inspection: <projectDir>/bin exists only on the head node, and rayapp unzips it 0644.
    plugin_src = os.path.join(_TEMPLATE, "nf-ray-plugin", "src", "main", "groovy")
    path = os.path.join(plugin_src, "ai", "anyscale", "nfray", "RayExecutor.groovy")
    with open(path) as handle:
        text = handle.read()
    assert re.search(r"@Override\s+Path getBinDir\(\)", text), "getBinDir() is not overridden"
    assert "stagedBinDir = stageBinDir()" in text, "register() no longer copies bin/"
    assert "getTempDir('bin')" in text, "bin/ is no longer copied under the work directory"
    assert "OWNER_EXECUTE" in text, "the copy of bin/ is no longer made executable"


@check("nextflow: numeric params are coerced before arithmetic")
def _() -> None:
    # CLI params arrive as Strings: Groovy's "4" - 1 is "4", and 0.."4" has 53 elements.
    for name, params in (
        ("pipeline/smoke.nf", ["shards"]),
        ("pipeline/main.nf", ["intervals"]),
    ):
        path = os.path.join(_TEMPLATE, name)
        with open(path) as handle:
            text = handle.read()
        assert "as Integer" in text, f"{name}: numeric params must be coerced"
        for param in params:
            assert f"params.{param}" in text, f"{name}: expected params.{param}"
        assert "params.shards - 1" not in text, f"{name}: string arithmetic on a param"
        assert "params.intervals - 1" not in text, f"{name}: string arithmetic on a param"


@check("nextflow: boolean params go through flag(), never straight into an if")
def _() -> None:
    # `--cnn false` arrives as the String "false", which Groovy treats as true.
    path = os.path.join(_TEMPLATE, "pipeline", "main.nf")
    with open(path) as handle:
        text = handle.read()
    coerced = re.findall(r"flag\(params\.(\w+)\)", text)
    assert "cnn" in coerced, "main.nf: params.cnn is not coerced"
    for param in coerced:
        raw = re.findall(rf"params\.{param}\b", text)
        assert len(raw) == 1, f"main.nf reads params.{param} raw {len(raw) - 1} more time(s)"


if _FAILURES:
    print(f"\n{len(_FAILURES)} failing: {', '.join(_FAILURES)}", file=sys.stderr)
    sys.exit(1)
print("\nall nf_ray unit checks passed")
