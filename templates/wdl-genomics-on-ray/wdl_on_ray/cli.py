"""``wdl-on-ray``: a thin wrapper that points miniwdl at the Ray backend.

Nothing here is required: ``miniwdl run --cfg ...`` with
``[scheduler] container_backend = ray`` does the same job, and that is the escape
hatch when you need a miniwdl feature this wrapper doesn't mention. What the
wrapper adds is the handful of defaults that are easy to get wrong and expensive
to get wrong quietly:

* selecting the ``ray`` backend;
* putting the run directory on shared storage, because the backend's filesystem
  contract requires it (see :mod:`wdl_on_ray.backend`);
* raising miniwdl's task concurrency to suit the *cluster*, not the driver node.
  Its default is the driver's ``nproc``, which silently caps a 500-core
  cluster at however many cores the head node happens to have.

Unrecognized arguments are forwarded to miniwdl verbatim, so
``wdl-on-ray run pipeline.wdl -i inputs.json --verbose`` works as expected.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from typing import Any

from wdl_on_ray import config as ray_config
from wdl_on_ray import job as ray_job
from wdl_on_ray import runtimes
from wdl_on_ray._version import __version__
from wdl_on_ray.backend import SHARED_STORAGE_PREFIXES

#: miniwdl's guideline ceiling for its own thread pool; beyond roughly this many
#: concurrent tasks the driver's Python process becomes the bottleneck.
MAX_TASK_CONCURRENCY = 200


def default_run_dir() -> str:
    """Pick a run directory that every node in the cluster can see."""
    override = os.environ.get("WDL_ON_RAY_RUN_DIR")
    if override:
        return override
    for prefix in SHARED_STORAGE_PREFIXES:
        if os.path.isdir(prefix) and os.access(prefix, os.W_OK):
            return os.path.join(prefix, "wdl-on-ray", "runs")
    return os.path.abspath("_wdl_runs")


def _setenv(key: str, value: str, *, force: bool = False) -> None:
    """Set a ``MINIWDL__*`` override without clobbering the caller's own."""
    if force or key not in os.environ:
        os.environ[key] = value


def _cluster_cpus() -> int:
    """Total cluster CPUs, or 0 if we can't ask without unwanted side effects.

    Goes through :func:`wdl_on_ray.backend.connect` instead of calling
    ``ray.init()`` here. That matters more than it looks: ``py_modules`` can only be
    attached when the connection is *established*, so an independent ``ray.init()``
    at this point, which is what this function used to do, left every Ray worker
    without ``wdl_on_ray`` on its import path, and the eventual failure named a
    deserialization error, which points nowhere near a thread pool.
    """
    import logging

    import ray
    from WDL.runtime.config import Loader

    from wdl_on_ray.backend import connect

    if not ray.is_initialized() and not os.environ.get("RAY_ADDRESS"):
        # No cluster to ask. Starting a local Ray instance merely to size a thread
        # pool would be a surprising side effect, and miniwdl's own default (the
        # driver's nproc) is already right for a single-node run.
        return 0
    try:
        connect(ray_config.load(Loader(logging.getLogger("wdl-on-ray"))), _quiet_logger())
        return int(ray.cluster_resources().get("CPU", 0))
    except Exception:
        return 0


def _quiet_logger() -> Any:
    """A logger with miniwdl's NOTICE level, for use before miniwdl's own setup.

    ``connect`` logs at NOTICE, which the stdlib does not define; miniwdl installs
    it at import time, so borrow that instead of reimplementing it.
    """
    import logging

    logger = logging.getLogger("wdl-on-ray.connect")
    if not hasattr(logger, "notice"):  # pragma: no cover - miniwdl adds this
        logger.notice = logger.info  # type: ignore[attr-defined]
    return logger


def _apply_run_defaults(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    """Translate our flags into miniwdl env overrides and argv."""
    _setenv("MINIWDL__SCHEDULER__CONTAINER_BACKEND", "ray", force=True)
    if args.container_runtime:
        _setenv("MINIWDL__RAY__CONTAINER_RUNTIME", args.container_runtime, force=True)
    if args.max_cpu:
        _setenv("MINIWDL__RAY__MAX_CPU", str(args.max_cpu), force=True)
    if args.ray_address:
        _setenv("MINIWDL__RAY__ADDRESS", args.ray_address, force=True)
    if args.tool_wheel_dir:
        _setenv("MINIWDL__RAY__TOOL_WHEEL_DIR", args.tool_wheel_dir, force=True)
    if args.call_cache:
        # miniwdl's call cache ships off (`put`/`get` both false) and defaults to a
        # node-local `~/.cache/miniwdl`, which on a multi-node cluster caches results
        # where the next task will not look for them. Both have to move together, so
        # this is one flag rather than three env vars.
        _setenv("MINIWDL__CALL_CACHE__DIR", args.call_cache, force=True)
        _setenv("MINIWDL__CALL_CACHE__PUT", "true", force=True)
        _setenv("MINIWDL__CALL_CACHE__GET", "true", force=True)

    if args.task_concurrency:
        concurrency = args.task_concurrency
    else:
        # 0 leaves miniwdl's own default (driver nproc) in place, which is right
        # for a single-node run and wrong for anything larger.
        concurrency = min(MAX_TASK_CONCURRENCY, _cluster_cpus())
    if concurrency > 0:
        _setenv("MINIWDL__SCHEDULER__TASK_CONCURRENCY", str(concurrency))

    argv = list(passthrough)
    if not any(a == "--dir" or a.startswith("--dir=") for a in argv):
        # No mkdir: miniwdl creates the run directory (and any missing parents)
        # itself. Creating it here was not only redundant, it made merely *computing*
        # the argv a filesystem write, which fails wherever the working directory
        # is read-only, as it is inside a Nix build sandbox.
        argv += ["--dir", args.dir or default_run_dir()]
    return argv


def _warn_missing_downloaders(argv: list[str]) -> None:
    """Report remote-input schemes this node cannot localize, before the run starts.

    miniwdl downloads a remote `File` input inside a synthesised WDL task that shells
    out to `aws`/`gsutil`/`aria2c`, expecting the binary from that task's container
    image. Under `none` and `native` there is no image, so the binary has to be on the
    node, and when it is not, the run fails with exit 127 from a task named something
    like `aws_s3_cp`, several directories deep, naming no scheme and no URI.

    `envs.missing_downloaders` has always been able to answer this; until now only
    `doctor` asked, which is the one moment nobody is about to hit the failure.
    """
    import json
    import pathlib

    from wdl_on_ray import envs

    # The URIs live in the inputs JSON and in bare `key=s3://...` arguments. Read
    # whatever is cheaply available and stay quiet if anything is unparseable: this
    # is a pre-flight courtesy, not a validator, and must never block a run.
    values: list[object] = [a for a in argv if "://" in a]
    for flag in ("-i", "--input"):
        if flag not in argv:
            continue
        try:
            path = argv[argv.index(flag) + 1]
            values.append(json.loads(pathlib.Path(path).read_text()))
        except (IndexError, OSError, ValueError):
            continue

    missing = envs.missing_downloaders(values)
    if missing:
        listed = ", ".join(f"{scheme}:// needs {exe}" for scheme, exe in sorted(missing.items()))
        print(
            f"warning: this node cannot localize some remote inputs ({listed}).\n"
            "         miniwdl downloads them inside a task, so this surfaces as exit 127\n"
            "         from a synthesised download task rather than as a clear error here.",
            file=sys.stderr,
        )


def _cmd_run(args: argparse.Namespace, passthrough: list[str]) -> int:
    import logging

    from WDL import CLI
    from WDL.runtime.config import Loader

    from wdl_on_ray.backend import register_backend

    argv = _apply_run_defaults(args, passthrough)
    _warn_missing_downloaders(argv)
    # Works whether or not this package is pip-installed; see register_backend.
    register_backend(Loader(logging.getLogger("wdl-on-ray")))
    return int(CLI.main(["run", *argv]) or 0)


def _cmd_check(args: argparse.Namespace, passthrough: list[str]) -> int:
    from WDL import CLI

    del args
    return int(CLI.main(["check", *passthrough]) or 0)


def _dist(name: str) -> str:
    """Installed version of a distribution, or ``MISSING``."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return "MISSING"


def _cmd_doctor(args: argparse.Namespace, passthrough: list[str]) -> int:
    """Report what the backend would decide, without running anything."""
    del args, passthrough
    import logging

    from WDL.runtime.config import Loader

    print(f"wdl-on-ray {__version__}")
    print(f"python      {sys.version.split()[0]}")
    for label in ("ray", "miniwdl"):
        print(f"{label:<11} {_dist(label)}")

    cfg = Loader(logging.getLogger("wdl-on-ray"))
    resolved = ray_config.load(cfg)

    print("\ncontainer runtimes on this node:")
    for name in (*runtimes.AUTO_ORDER, "none", "native", "ray"):
        runtime = runtimes.get(name)
        exe = runtime.default_exe
        if runtime.env_is_image:
            # Ray starts the nested container itself, so there is no executable to
            # probe. What decides whether this mode can run is the image map, and
            # whether images have been built against this cluster's versions.
            mapped = len(resolved.task_image_map)
            state = (
                f"{mapped} image(s) mapped" if mapped
                else ("no images mapped; unmapped tasks run in the cluster image"
                      if resolved.task_image_fallback == "cluster"
                      else "NOT USABLE: [ray] task_image_map is empty")
            )
            print(f"  {name:<12} {state}")
            print(f"  {'':<12} task images must be built on ray {_dist('ray')} / "
                  f"python {sys.version.split()[0]} exactly")
            continue
        if runtime.provides_env:
            # Not "always available" like `none`: this one needs Ray's env plugins to
            # work, and those have prerequisites better reported before a run rather
            # than discovering them when the first task dispatches.
            missing = [n for n in ("virtualenv", "pip") if not importlib.util.find_spec(n)]
            state = f"MISSING {', '.join(missing)} in this Python" if missing else "ready"
            print(f"  {name:<12} {state} (Ray supplies each task's tools)")
            continue
        if not runtime.isolated:
            print(f"  {name:<12} always available (no nested container)")
            continue
        found = shutil.which(exe[0]) if exe else None
        print(f"  {name:<12} {found or 'not found'}")

    if resolved.task_image_map:
        print("\n[ray] task_image_map (runtime.docker -> what container_runtime=ray runs):")
        for declared, actual in sorted(resolved.task_image_map.items()):
            print(f"  {declared}")
            print(f"    -> {actual}")

    # Which remote input schemes this node can actually localize. Not the pipeline's tools:
    # miniwdl downloads a remote input with a synthesised WDL task that shells out to these,
    # expecting the binary from that task's container image, so under `native`, with no
    # container, the node has to have them. Reported here because nothing declares them.
    from wdl_on_ray import envs

    print("\ninput downloaders (miniwdl shells out to these; native mode needs them on PATH):")
    for scheme, executable in sorted(envs.DOWNLOADER_EXECUTABLES.items()):
        found = shutil.which(executable)
        print(f"  {scheme + '://':<10} {executable:<8} {found or 'not found'}")

    print("\nresolved [ray] configuration:")
    for field, value in sorted(vars(resolved).items()):
        print(f"  {field:<22} {value!r}")

    run_dir = default_run_dir()
    shared = any(os.path.abspath(run_dir).startswith(p) for p in SHARED_STORAGE_PREFIXES)
    print(f"\ndefault run dir  {run_dir}")
    print(f"  on shared storage: {'yes' if shared else 'NO (single-node runs only)'}")

    # Reported because it is off unless asked for, and because a long assembly that
    # dies without it starts again from nothing.
    cache_on = cfg["call_cache"].get_bool("get") and cfg["call_cache"].get_bool("put")
    cache_dir = cfg["call_cache"]["dir"] if cache_on else None
    print(f"\ncall cache       {'on' if cache_on else 'off (--call-cache DIR turns it on)'}")
    if cache_dir:
        on_shared = any(os.path.abspath(cache_dir).startswith(p) for p in SHARED_STORAGE_PREFIXES)
        print(f"  dir            {cache_dir}")
        print(f"  on shared storage: {'yes' if on_shared else 'NO (other nodes will miss it)'}")

    try:
        import ray

        if ray.is_initialized() or os.environ.get("RAY_ADDRESS"):
            cpus = _cluster_cpus()
            print(f"\ncluster CPUs     {cpus or 'unknown'}")
        else:
            print("\ncluster          not connected (RAY_ADDRESS unset; a local Ray")
            print("                 instance will be started on demand)")
    except ImportError:
        pass
    return 0


def _cmd_probe_image(args: argparse.Namespace, passthrough: list[str]) -> int:
    """Run one Ray task in a candidate image and report whether the mode can work."""
    del passthrough
    import logging

    import ray

    from WDL.runtime.config import Loader

    resolved = ray_config.load(Loader(logging.getLogger("wdl-on-ray")))
    run_dir = args.dir or default_run_dir()
    os.makedirs(run_dir, exist_ok=True)

    shared = any(os.path.abspath(run_dir).startswith(p) for p in SHARED_STORAGE_PREFIXES)
    print(f"probing   {args.image}")
    print(f"run dir   {run_dir}" + ("" if shared else "   (NOT on shared storage)"))

    ray.init(address=resolved.address, namespace=resolved.namespace, ignore_reinit_error=True)
    driver = {"python": sys.version.split()[0], "ray": ray.__version__}
    print(f"driver    ray {driver['ray']}, python {driver['python']}\n")

    # Ship the probe's code inside the task. Without this cloudpickle sends a module
    # path, and the image being probed would have to import wdl_on_ray (and, through
    # it, miniwdl) before it could answer whether it can run a task at all -- so a
    # perfectly good task image would fail deserialization and be reported unusable.
    # Same mechanism the real dispatch path uses; see backend._pickle_worker_modules_by_value.
    from wdl_on_ray.backend import _pickle_worker_modules_by_value

    _pickle_worker_modules_by_value()

    marker = f"wdl-on-ray probe {os.getpid()}"
    remote = ray.remote(ray_job.probe_image).options(
        num_cpus=1,
        runtime_env={"image_uri": args.image},
        max_retries=0,
    )
    try:
        got = ray.get(remote.remote(run_dir, marker), timeout=args.timeout)
    except Exception as exn:  # noqa: BLE001 - the failure itself is the report
        print(f"FAILED to run a task in that image:\n  {type(exn).__name__}: {exn}")
        print(
            "\nCommon causes: the image is unreachable from the workers; its Ray or Python"
            "\nversion does not match the driver's; or the cloud requires the ray container"
            "\nto run privileged for nested containers (Kubernetes-backed clouds do)."
        )
        return 1

    print(f"task      ray {got['ray']}, python {got['python']}  on {got['host']}")
    problems = []
    if got["ray"] != driver["ray"]:
        problems.append(f"ray version differs: driver {driver['ray']}, image {got['ray']}")
    if got["python"] != driver["python"]:
        problems.append(
            f"python version differs: driver {driver['python']}, image {got['python']}"
            " (Ray requires an exact match, patch level included)"
        )
    for label, key in (
        ("run directory not visible inside the image", "run_dir_visible"),
        ("run directory not writable inside the image", "run_dir_writable"),
        ("wrote to the run directory but read back wrong content", "readback_ok"),
    ):
        if not got[key]:
            problems.append(label)
    if got["error"]:
        problems.append(got["error"])

    print()
    if problems:
        for problem in problems:
            print(f"  FAIL  {problem}")
        print(
            "\ncontainer_runtime=ray will not work with this image as configured."
            "\nA run directory that is invisible inside the image is fatal to the mode:"
            "\nminiwdl passes files between tasks by path."
        )
        return 1

    print("  OK    versions match and the shared run directory is readable and writable")
    print("\ncontainer_runtime=ray can use this image. Add it to [ray] task_image_map.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wdl-on-ray",
        description="Run WDL pipelines on a Ray or Anyscale cluster.",
        epilog=(
            "Unrecognized arguments are passed through to miniwdl, e.g."
            " `wdl-on-ray run p.wdl -i inputs.json --verbose`."
        ),
    )
    parser.add_argument("--version", action="version", version=f"wdl-on-ray {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run a WDL workflow on Ray")
    run.add_argument(
        "--container-runtime",
        choices=list(ray_config.CONTAINER_RUNTIMES),
        help="how each task's container is run on the worker (default: auto)",
    )
    run.add_argument(
        "--dir",
        help=f"run directory; must be visible from every node (default: {default_run_dir()})",
    )
    run.add_argument(
        "--task-concurrency",
        type=int,
        metavar="N",
        help=f"max WDL tasks in flight (default: min(cluster CPUs, {MAX_TASK_CONCURRENCY}))",
    )
    run.add_argument(
        "--max-cpu",
        type=int,
        metavar="N",
        help="per-task CPU ceiling; set this when the cluster autoscales to nodes"
        " larger than any currently running",
    )
    run.add_argument(
        "--tool-wheel-dir",
        metavar="PATH_OR_URL",
        help="where --container-runtime native finds the tool wheels: a directory on shared"
        " storage or a published --find-links index (default: resolve from a package index)",
    )
    run.add_argument(
        "--call-cache",
        metavar="DIR",
        help="reuse completed tasks across runs, caching to DIR; must be visible from"
        " every node, and must outlive the cluster to survive a job retry"
        " (default: off, as in miniwdl)",
    )
    run.add_argument("--ray-address", help="Ray cluster address (default: auto)")
    run.set_defaults(func=_cmd_run)

    check = subparsers.add_parser("check", help="type-check a WDL document (miniwdl check)")
    check.set_defaults(func=_cmd_check)

    doctor = subparsers.add_parser(
        "doctor", help="report the environment and the configuration that would be used"
    )
    doctor.set_defaults(func=_cmd_doctor)

    probe = subparsers.add_parser(
        "probe-image",
        help="check whether an image can be used with --container-runtime ray",
        description="Runs one Ray task inside IMAGE and reports whether its Ray and Python"
        " versions match this driver's and whether the shared run directory is readable and"
        " writable from inside it. Both are preconditions for --container-runtime ray.",
    )
    probe.add_argument("image", metavar="IMAGE", help="image URI to test, as Ray's image_uri")
    probe.add_argument("--dir", help=f"run directory to test (default: {default_run_dir()})")
    probe.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        metavar="SECONDS",
        help="how long to wait for the probe task, which includes pulling the image"
        " on a cold node (default: 600)",
    )
    probe.set_defaults(func=_cmd_probe_image)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)
    func: Any = args.func
    return int(func(args, passthrough))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
