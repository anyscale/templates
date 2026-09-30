"""``wdl-on-ray``: a thin miniwdl wrapper that selects the Ray backend and sets run defaults."""

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

#: miniwdl's guideline ceiling for its thread pool; past it the driver process is the bottleneck.
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
    if force or key not in os.environ:
        os.environ[key] = value


def _find_cluster() -> str | None:
    # Only looks: sizing a thread pool is no reason to start or join Ray.
    import logging

    from WDL.runtime.config import Loader

    from wdl_on_ray.backend import find_cluster

    return find_cluster(ray_config.load(Loader(logging.getLogger("wdl-on-ray"))))


def _quiet_logger() -> Any:
    import logging

    logger = logging.getLogger("wdl-on-ray.connect")
    if not hasattr(logger, "notice"):  # pragma: no cover - miniwdl adds this
        logger.notice = logger.info  # type: ignore[attr-defined]
    return logger


def _apply_run_defaults(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
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
        # miniwdl's cache ships off and node-local; dir, put and get have to move together.
        _setenv("MINIWDL__CALL_CACHE__DIR", args.call_cache, force=True)
        _setenv("MINIWDL__CALL_CACHE__PUT", "true", force=True)
        _setenv("MINIWDL__CALL_CACHE__GET", "true", force=True)

    if args.task_concurrency:
        _setenv("MINIWDL__SCHEDULER__TASK_CONCURRENCY", str(args.task_concurrency), force=True)
    elif _find_cluster():
        # Not sized to the nodes up now: the autoscaler scales on Ray's queue of what doesn't fit.
        # Not forced, so the caller's own MINIWDL__SCHEDULER__TASK_CONCURRENCY stays.
        _setenv("MINIWDL__SCHEDULER__TASK_CONCURRENCY", str(MAX_TASK_CONCURRENCY))

    argv = list(passthrough)
    if not any(a == "--dir" or a.startswith("--dir=") for a in argv):
        # No mkdir: miniwdl creates it, and building argv must not write to a read-only cwd.
        argv += ["--dir", args.dir or default_run_dir()]
    return argv


def _warn_missing_downloaders(argv: list[str]) -> None:
    import json
    import pathlib

    from wdl_on_ray import envs

    # Best effort, from the inputs JSON and bare key=uri arguments; must never block a run.
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
    # Needed when this package is not installed.
    register_backend(Loader(logging.getLogger("wdl-on-ray")))
    return int(CLI.main(["run", *argv]) or 0)


def _cmd_check(args: argparse.Namespace, passthrough: list[str]) -> int:
    from WDL import CLI

    del args
    return int(CLI.main(["check", *passthrough]) or 0)


def _dist(name: str) -> str:
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
            # Ray starts the nested container, so there is no executable to probe.
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
            # Needs Ray's env plugins, whose prerequisites otherwise fail at first dispatch.
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

    cache_on = cfg["call_cache"].get_bool("get") and cfg["call_cache"].get_bool("put")
    cache_dir = cfg["call_cache"]["dir"] if cache_on else None
    print(f"\ncall cache       {'on' if cache_on else 'off (--call-cache DIR turns it on)'}")
    if cache_dir:
        on_shared = any(os.path.abspath(cache_dir).startswith(p) for p in SHARED_STORAGE_PREFIXES)
        print(f"  dir            {cache_dir}")
        print(f"  on shared storage: {'yes' if on_shared else 'NO (other nodes will miss it)'}")

    _print_cluster(resolved)
    return 0


def _print_cluster(resolved: ray_config.RayConfig) -> None:
    # Joined only if found: joining nothing would start a local instance.
    from wdl_on_ray import backend

    address = backend.find_cluster(resolved)
    own = os.environ.get("MINIWDL__SCHEDULER__TASK_CONCURRENCY")
    if own:
        pool = f"{own} (MINIWDL__SCHEDULER__TASK_CONCURRENCY)"
    elif address:
        pool = f"{MAX_TASK_CONCURRENCY} (--task-concurrency N changes it)"
    else:
        pool = f"miniwdl's default, this node's {os.cpu_count()} cores"
    if address is None:
        print("\ncluster          none found: a run starts a local Ray instance on this node")
        print("                 (looked for [ray] address, RAY_ADDRESS, then")
        print(f"                 {backend._ray_address_file()})")
        print(f"  task pool      {pool}")
        return

    print(f"\ncluster          {address}")
    try:
        import ray

        backend.connect(resolved, _quiet_logger())
        found = backend.RayContainer._probe_limits(resolved, _quiet_logger())
    except Exception as exn:  # noqa: BLE001 - the failure itself is the report
        print(f"  not reachable  {type(exn).__name__}: {exn}")
        return
    print(f"  CPUs           {int(ray.cluster_resources().get('CPU', 0))}")
    print(f"  task ceiling   {_describe_ceiling(backend.limits_from(found, resolved), found)}")
    print(f"  task pool      {pool}")


def _describe_ceiling(limits: dict[str, int], found: tuple[float, float] | None) -> str:
    from wdl_on_ray.backend import NO_LIMIT

    cpu, mem = limits["cpu"], limits["mem_bytes"]
    text = ", ".join(
        (
            "CPU unclamped" if cpu == NO_LIMIT else f"{cpu} CPU",
            "memory unclamped" if mem == NO_LIMIT else f"{mem / 2**30:.1f} GiB",
        )
    )
    if found is None:
        text += " (no node that can run tasks is up to measure)"
    return text


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

    # By value, so the probed image need not import wdl_on_ray (or miniwdl) to answer.
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
        help=f"max WDL tasks in flight (default: {MAX_TASK_CONCURRENCY} on a Ray cluster, which"
        " queues what does not fit; with none to join, miniwdl's default, this node's cores)",
    )
    run.add_argument(
        "--max-cpu",
        type=int,
        metavar="N",
        help="per-task CPU ceiling (default: the largest worker up at startup, and none"
        " if no worker is up yet); set it to the worker shape if they may not be up",
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
