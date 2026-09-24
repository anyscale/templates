"""Build the Ray ``runtime_env`` for one task.

Most Nextflow pipelines need nothing here: the template's image carries the
toolchain, every process finds its tools on ``PATH``, and ``runtime_env`` stays
empty. Two cases need more.

**A pipeline that declares per-process containers.** Every nf-core module has a
``container 'quay.io/biocontainers/...'`` directive. Nextflow resolves those
itself when ``docker.enabled`` or ``apptainer.enabled`` is set -- it writes the
container invocation into ``.command.run``, so it never reaches this executor.
Inside a Ray worker, though, there is no container runtime to nest into, so that
route is closed and the pipeline needs either ``-profile conda`` (Nextflow builds
each process an environment; point ``NXF_CONDA_CACHEDIR`` at shared storage and it
is built once per cluster) or the next case.

**A pipeline that wants a real per-process image.** Ray can start the *worker
process itself* inside an image via ``runtime_env={'image_uri': ...}``, which is
the closest thing to per-process containers that works here. It comes with a hard
constraint: Ray extracts its own and Python's version from the candidate image and
refuses anything that does not match the cluster to the patch. A stock
biocontainer will therefore never work verbatim -- it has no Ray in it at all.

So ``ext.image`` is mapped, not used: :attr:`nf_ray.config.Config.image_map` takes
the image the pipeline declared to one rebuilt on a matching base. The default for
an unmapped image is to **fail**, not to quietly run in the cluster's image,
because a task that silently runs against different software than it declared
produces results nobody can reproduce. ``nf-ray probe-image`` checks a candidate
before a pipeline depends on it.
"""

from __future__ import annotations

from typing import Any

from nf_ray.config import Config
from nf_ray.directives import Directives


class ImageUnavailable(RuntimeError):
    """``ext.image`` named an image that cannot be launched on this cluster."""


def resolve_image(declared: str, config: Config) -> str:
    """Map a declared ``ext.image`` onto one launchable on this cluster.

    Returns ``""`` when the task should run in the cluster's own image.
    """
    if not declared:
        return ""

    mapped = config.image_map.get(declared) or config.image_map.get("*")
    if mapped:
        return mapped

    if config.image_fallback == "ignore":
        return ""

    raise ImageUnavailable(
        f"ext.image = {declared!r} has no entry in NF_RAY_IMAGE_MAP.\n"
        "Ray starts the worker process inside the image and rejects any image whose Ray\n"
        "and Python versions do not match this cluster's to the patch, so a stock\n"
        "biocontainer cannot be used verbatim -- it contains no Ray at all.\n"
        "Either map it to an image rebuilt on a matching base:\n"
        '  env { NF_RAY_IMAGE_MAP = \'{"%s": "<your-rebuilt-image>"}\' }\n'
        "or drop the declaration and let the process use the cluster's image:\n"
        "  env { NF_RAY_IMAGE_FALLBACK = 'ignore' }\n"
        "Check a candidate first with:  nf-ray probe-image <uri>" % declared
    )


def build_runtime_env(directives: Directives, config: Config) -> dict[str, Any]:
    """The ``runtime_env`` for one task, or ``{}`` for none.

    A per-task ``runtime_env`` is not free -- Ray sets one up per distinct env per
    node -- so an empty dict is returned rather than an empty-but-present one,
    which Ray treats as a real (if trivial) environment to install.
    """
    env: dict[str, Any] = dict(config.runtime_env)

    image = resolve_image(directives.image, config)
    if image:
        env["image_uri"] = image

    return env
