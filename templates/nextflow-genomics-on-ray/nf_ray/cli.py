"""``nf-ray``, the CLI the executor plugin drives. RayExecutor.groovy parses the output of
submit/status/kill, which run per task and so must not import Ray: handlers import lazily.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

from nf_ray._version import __version__
from nf_ray.config import Config, is_shared_storage, shared_storage_warning


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--socket",
        default="",
        help="daemon socket (default: /tmp/nf-ray-<workDir hash>.sock, or NF_RAY_SOCKET)",
    )
    parser.add_argument(
        "--work-dir",
        default="",
        help="Nextflow workDir (default: cwd, or NF_RAY_WORK_DIR)",
    )


def _config(args: argparse.Namespace) -> Config:
    if getattr(args, "work_dir", ""):
        os.environ["NF_RAY_WORK_DIR"] = args.work_dir
    if getattr(args, "socket", ""):
        os.environ["NF_RAY_SOCKET"] = args.socket
    return Config.from_env()


def cmd_submit(args: argparse.Namespace) -> int:
    """Submit one job script. Prints the line ``parseJobId`` matches."""
    from nf_ray import client  # noqa: PLC0415

    config = _config(args)
    # AbstractGridExecutor runs submit in the task's work directory, as with sbatch.
    work_dir = os.getcwd()
    client.ensure_daemon(config.socket_path, config.work_dir)
    task_id = client.submit(config.socket_path, args.script, work_dir)
    print(f"Submitted ray task {task_id}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Print the whole queue, one ``<id> <state>`` per line."""
    from nf_ray import client  # noqa: PLC0415

    config = _config(args)
    if not client.ping(config.socket_path):
        # No daemon means no tasks: an empty queue, not an error before the first submit.
        return 0
    for task_id, state in client.status(config.socket_path):
        print(f"{task_id} {state}")
    return 0


def cmd_kill(args: argparse.Namespace) -> int:
    from nf_ray import client  # noqa: PLC0415

    config = _config(args)
    if client.ping(config.socket_path):
        client.kill(config.socket_path, [int(i) for i in args.ids])
    return 0


def cmd_daemon(args: argparse.Namespace) -> int:
    """Run the daemon in the foreground. Normally started by ``submit``."""
    from nf_ray import daemon  # noqa: PLC0415 -- pulls in Ray; only this path needs it

    daemon.serve(_config(args))
    return 0


def cmd_up(args: argparse.Namespace) -> int:
    from nf_ray import client  # noqa: PLC0415

    config = _config(args)
    client.ensure_daemon(config.socket_path, config.work_dir)
    print(f"nf-ray daemon listening on {config.socket_path}")
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    from nf_ray import client  # noqa: PLC0415

    config = _config(args)
    client.shutdown(config.socket_path)
    print("nf-ray daemon stopped")
    return 0


# A missing tool otherwise surfaces as a task exiting 127 deep inside a scatter.
EXPECTED_TOOLS = (
    "nextflow",
    "java",
    "bwa-mem2",
    "samtools",
    "bcftools",
    "fastp",
    "gatk",
    "rtg",
    "multiqc",
)


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report the executor's configuration and this node's environment, running nothing."""
    # Reads NF_RAY_* only; the ray {} scope arrives via the plugin, so a doctor by hand misses it.
    config = _config(args)
    problems: list[str] = []

    print(f"nf-ray {__version__}")
    print(f"  python           {'.'.join(str(v) for v in sys.version_info[:3])}")
    if sys.version_info[:2] != (3, 12):
        problems.append(
            f"python is {'.'.join(str(v) for v in sys.version_info[:2])}, "
            "but this template's image ships 3.12; a Ray driver and its workers "
            "must agree on the interpreter"
        )

    try:
        import ray  # noqa: PLC0415 -- doctor reports a missing Ray, so it must not fail at import

        print(f"  ray              {ray.__version__}")
        if not ray.__version__.startswith("2.58."):
            problems.append(
                f"ray is {ray.__version__}, but this template is built and tested "
                "against 2.58.0; `ext.image` in particular requires an exact match"
            )
    except ImportError:
        problems.append("ray is not importable in this interpreter")

    print(f"  workDir          {config.work_dir}")
    print(f"  socket           {config.socket_path}")
    print(f"  shared storage   {'yes' if is_shared_storage(config.work_dir) else 'NO'}")
    warning = shared_storage_warning(config.work_dir)
    if warning:
        problems.append(warning)

    print(f"  scheduling       {config.scheduling_strategy}")
    print(f"  ray max_retries  {config.task_max_retries} (Nextflow owns retry policy)")
    print(f"  clamp resources  {config.clamp_resources}")
    print(f"  image fallback   {config.image_fallback}")
    if config.image_map:
        for declared, mapped in config.image_map.items():
            print(f"  image map        {declared} -> {mapped}")
    if config.extra_resources:
        print(f"  extra resources  {json.dumps(config.extra_resources)}")

    ceiling = (config.max_node_cpus, config.max_node_memory_gb, config.max_node_gpus)
    if any(ceiling):
        print(
            f"  declared ceiling {config.max_node_cpus:g} CPU / "
            f"{config.max_node_memory_gb:g} GiB / {config.max_node_gpus:g} GPU per node"
        )
    else:
        print("  declared ceiling none -- falls back to the largest node seen alive")

    print("  tools on PATH")
    for tool in EXPECTED_TOOLS:
        found = shutil.which(tool)
        print(f"    {tool:<12} {found or 'MISSING'}")
        if not found:
            problems.append(f"{tool} is not on PATH")

    from nf_ray import client  # noqa: PLC0415

    if client.ping(config.socket_path):
        info = client.info(config.socket_path)
        print("  daemon           running")
        print(f"    nodes alive    {info['nodes_alive']}")
        print(f"    node ceiling   {json.dumps(info['limits'])}")
        print(f"    task counts    {json.dumps(info['counts'])}")
    else:
        print("  daemon           not running (started on first submit)")

    if problems:
        print(f"\n{len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("\nno problems found")
    return 0


def cmd_probe_image(args: argparse.Namespace) -> int:
    """Run one task inside a candidate ``ext.image`` and report what it sees."""
    import ray  # noqa: PLC0415 -- the hot subcommands must not import Ray

    from nf_ray import daemon, job  # noqa: PLC0415

    config = _config(args)
    # Via the daemon's connect logic: off a cluster it starts a local Ray instead of failing.
    scheduler = daemon.Scheduler(config)
    try:
        remote = ray.remote(job.probe).options(  # type: ignore[arg-type]
            num_cpus=1,
            runtime_env={"image_uri": args.image},
            max_retries=0,
        )
        result = ray.get(remote.remote(config.work_dir, f"nf-ray probe {os.getpid()}"))
    except Exception as exn:  # noqa: BLE001
        print(f"{args.image}\n  UNUSABLE: {type(exn).__name__}: {exn}", file=sys.stderr)
        print(
            "\nRay extracts its own and Python's version from the image and refuses a\n"
            "mismatch, so a stock biocontainer will always fail here. Rebuild the tool\n"
            "on top of anyscale/ray:2.58.0-py312-cu128 and probe that instead.",
            file=sys.stderr,
        )
        return 1
    finally:
        scheduler.shutdown()

    print(f"{args.image}")
    for key in sorted(result):
        print(f"  {key:<18} {result[key]}")

    ok = bool(result.get("work_dir_visible")) and bool(result.get("readback_ok"))
    if not ok:
        print(
            "\nUNUSABLE: the image cannot read and write the work directory at the same\n"
            "absolute path the driver uses. Nextflow stages inputs by path, so a process\n"
            "in this image could not consume the previous process's output.",
            file=sys.stderr,
        )
        return 1
    print("\nusable")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nf-ray",
        description="Ray backend for the Nextflow 'ray' executor.",
    )
    parser.add_argument("--version", action="version", version=f"nf-ray {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("submit", help="submit one Nextflow job script")
    p.add_argument("script", help="job script, usually .command.run")
    _add_common(p)
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("status", help="print the queue as '<id> <state>' lines")
    _add_common(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("kill", help="cancel tasks by id")
    p.add_argument("ids", nargs="+")
    _add_common(p)
    p.set_defaults(func=cmd_kill)

    p = sub.add_parser("daemon", help="run the daemon in the foreground")
    _add_common(p)
    p.set_defaults(func=cmd_daemon)

    p = sub.add_parser("up", help="start the daemon if it is not running")
    _add_common(p)
    p.set_defaults(func=cmd_up)

    p = sub.add_parser("down", help="stop the daemon")
    _add_common(p)
    p.set_defaults(func=cmd_down)

    p = sub.add_parser("doctor", help="report the executor's configuration and environment")
    _add_common(p)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("probe-image", help="check an ext.image candidate on this cluster")
    p.add_argument("image")
    _add_common(p)
    p.set_defaults(func=cmd_probe_image)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exn:  # noqa: BLE001
        # Nextflow shows a failed submit's stderr: keep it to the message unless asked.
        if os.environ.get("NF_RAY_TRACEBACK"):
            raise
        print(f"nf-ray: {type(exn).__name__}: {exn}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
