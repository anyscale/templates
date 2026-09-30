"""Build the Ray ``runtime_env`` for one task."""

from __future__ import annotations

from typing import Any

from nf_ray.config import Config
from nf_ray.directives import Directives


class ImageUnavailable(RuntimeError):
    """``ext.image`` named an image that cannot be launched on this cluster."""


def resolve_image(declared: str, config: Config) -> str:
    """Map a declared ``ext.image`` onto a launchable one; ``""`` means the cluster's image."""
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
    """The ``runtime_env`` for one task, or ``{}`` for none."""
    env: dict[str, Any] = dict(config.runtime_env)

    image = resolve_image(directives.image, config)
    if image:
        env["image_uri"] = image

    return env
