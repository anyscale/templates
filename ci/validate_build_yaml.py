#!/usr/bin/env python3
"""Validate BUILD.yaml: schema, paths, name uniqueness, and (optionally) GCP image accessibility."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, List, Optional

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)


REPO_ROOT = Path(__file__).resolve().parent.parent

ACCEPT = ", ".join([
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
])


# ---------------------------------------------------------------- schema

class Strict(BaseModel):
    """Reject unknown keys at every level."""
    model_config = ConfigDict(extra="forbid")


class Byod(Strict):
    docker_image: str
    ray_version: str


class ClusterEnv(Strict):
    image_uri: Optional[str] = Field(default=None, pattern=r"^anyscale/")
    byod: Optional[Byod] = None

    @model_validator(mode="after")
    def xor_image_or_byod(self):
        has_image, has_byod = self.image_uri is not None, self.byod is not None
        if has_image and has_byod:
            raise ValueError(
                "cluster_env has both 'image_uri' and 'byod'; pick exactly one"
            )
        if not has_image and not has_byod:
            raise ValueError(
                "cluster_env has neither 'image_uri' nor 'byod'; set exactly one"
            )
        return self


class ComputeConfig(Strict):
    GCP: str = Field(pattern=r"^configs/.+/gce\.yaml$")
    AWS: str = Field(pattern=r"^configs/.+/aws\.yaml$")


class Test(Strict):
    tests_path: str = Field(pattern=r"^tests/")
    command: str = Field(min_length=1)
    timeout_in_sec: int = Field(gt=0)


class Entry(Strict):
    name: str = Field(pattern=r"^[a-z0-9_-]+$")
    dir: str = Field(pattern=r"^templates/")
    cluster_env: ClusterEnv
    compute_config: ComputeConfig
    test: Test


# Schemas for the legacy compute config, applied to BUILD.yaml-referenced
# files. All fields are optional (lax) but extra keys are rejected (strict)
# — extra keys usually indicate the file is using the new schema.
#   ComputeTemplateConfig: https://docs.anyscale.com/ref/0.26.64/compute-config-api#computetemplateconfig-legacy
#   ComputeNodeType:       https://docs.anyscale.com/ref/0.26.64/compute-config-api#computenodetype-legacy
#   WorkerNodeType:        https://docs.anyscale.com/ref/0.26.64/other#workernodetype-legacy
#   Resources:             https://docs.anyscale.com/ref/0.26.64/other#resources-legacy

class LegacyResources(Strict):
    cpu: Optional[Any] = None
    gpu: Optional[Any] = None
    memory: Optional[Any] = None
    object_store_memory: Optional[Any] = None
    custom_resources: Optional[Any] = None


class LegacyComputeNodeType(Strict):
    """Used for head_node_type."""
    name: Optional[Any] = None
    instance_type: Optional[Any] = None
    resources: Optional[LegacyResources] = None
    labels: Optional[Any] = None
    aws_advanced_configurations_json: Optional[Any] = None
    gcp_advanced_configurations_json: Optional[Any] = None
    advanced_configurations_json: Optional[Any] = None
    flags: Optional[Any] = None


class LegacyWorkerNodeType(LegacyComputeNodeType):
    """Used for worker_node_types entries. Same as ComputeNodeType plus
    worker-specific scaling / spot fields."""
    min_workers: Optional[Any] = None
    max_workers: Optional[Any] = None
    use_spot: Optional[Any] = None
    fallback_to_ondemand: Optional[Any] = None


class LegacyComputeTemplateConfig(Strict):
    cloud_id: Optional[Any] = None
    maximum_uptime_minutes: Optional[Any] = None
    deployment_configs: Optional[Any] = None
    max_workers: Optional[Any] = None
    region: Optional[Any] = None
    allowed_azs: Optional[Any] = None
    head_node_type: Optional[LegacyComputeNodeType] = None
    worker_node_types: Optional[List[LegacyWorkerNodeType]] = None
    aws_advanced_configurations_json: Optional[Any] = None
    gcp_advanced_configurations_json: Optional[Any] = None
    advanced_configurations_json: Optional[Any] = None
    auto_select_worker_config: Optional[Any] = None
    flags: Optional[Any] = None
    idle_termination_minutes: Optional[Any] = None


# ------------------------------------------- filesystem + name uniqueness

def check_filesystem_and_uniqueness(entries: list[Entry]) -> list[str]:
    """Verify referenced paths exist, fields are unique across entries, and
    each entry's GCP/AWS configs differ. (Pydantic can't express filesystem
    state or cross-entry constraints, so we do this in Python.)"""
    seen_names: set[str] = set()
    seen_dirs: set[str] = set()
    seen_tests_paths: set[str] = set()
    errors: list[str] = []
    for e in entries:
        if e.name in seen_names:
            errors.append(f"duplicate name: {e.name!r}")
        seen_names.add(e.name)

        if e.dir in seen_dirs:
            errors.append(f"{e.name}: duplicate dir: {e.dir}")
        seen_dirs.add(e.dir)

        if e.test.tests_path in seen_tests_paths:
            errors.append(f"{e.name}: duplicate tests_path: {e.test.tests_path}")
        seen_tests_paths.add(e.test.tests_path)

        # tests_path basename must equal the entry's name. Catches stale
        # tests_path values when an entry is renamed.
        tests_basename = Path(e.test.tests_path.rstrip("/")).name
        if tests_basename != e.name:
            errors.append(
                f"{e.name}.test.tests_path: basename {tests_basename!r} must "
                f"equal name {e.name!r} (expected tests/{e.name}/)"
            )

        # GCP and AWS configs must live in the same directory under configs/.
        # Catches one-cloud-customized-but-not-the-other mistakes.
        gcp_parent = Path(e.compute_config.GCP).parent
        aws_parent = Path(e.compute_config.AWS).parent
        if gcp_parent != aws_parent:
            errors.append(
                f"{e.name}.compute_config: GCP and AWS configs must live in "
                f"the same `configs/<name>/` directory; got {gcp_parent} and "
                f"{aws_parent}"
            )

        # Custom compute config dirs must be named after the entry (the
        # `name` field is the source of truth across templates/, tests/,
        # and configs/). Shared `configs/basic-single-node/` is exempt.
        cfg_dir_basename = gcp_parent.name
        if (
            gcp_parent == aws_parent
            and cfg_dir_basename != "basic-single-node"
            and cfg_dir_basename != e.name
        ):
            errors.append(
                f"{e.name}.compute_config: dir basename {cfg_dir_basename!r} "
                f"must equal name {e.name!r} (expected configs/{e.name}/) — "
                f"or use the shared configs/basic-single-node/"
            )

        if not (REPO_ROOT / e.dir).is_dir():
            errors.append(f"{e.name}: dir not found: {e.dir}")
        for cloud in ("GCP", "AWS"):
            path = getattr(e.compute_config, cloud)
            if not (REPO_ROOT / path).is_file():
                errors.append(f"{e.name}.compute_config.{cloud}: not found: {path}")
        if not (REPO_ROOT / e.test.tests_path).is_dir():
            errors.append(f"{e.name}.test.tests_path: not found: {e.test.tests_path}")
        elif not (REPO_ROOT / e.test.tests_path / "tests.sh").is_file():
            errors.append(f"{e.name}.test.tests_path: missing tests.sh in {e.test.tests_path}")
    return errors


# ----------------------------------- redundant compute configs (warning)

BASIC_CONFIGS = (
    "configs/basic-single-node/aws.yaml",
    "configs/basic-single-node/gce.yaml",
)


def check_redundant_compute_configs(entries: list[Entry]) -> list[str]:
    """Warn when a custom compute config file is byte-equal to one of the
    shared `basic-single-node` configs. The author can just reference the
    shared one."""
    warnings: list[str] = []
    basics: dict[str, bytes] = {}
    for p in BASIC_CONFIGS:
        full = REPO_ROOT / p
        if full.is_file():
            basics[p] = full.read_bytes()

    for e in entries:
        for cloud in ("GCP", "AWS"):
            path = getattr(e.compute_config, cloud)
            if path in BASIC_CONFIGS:
                continue
            full = REPO_ROOT / path
            if not full.is_file():
                continue
            content = full.read_bytes()
            for basic_path, basic_bytes in basics.items():
                if content == basic_bytes:
                    warnings.append(
                        f"{e.name}.compute_config.{cloud}: {path} is byte-identical "
                        f"to {basic_path}; consider referencing the shared config "
                        f"instead of duplicating it"
                    )
                    break  # one warning per cloud is enough
    return warnings


# ----------------------------------------- compute config legacy schema

LEGACY_DOCS = (
    "https://docs.anyscale.com/ref/0.26.64/compute-config-api"
    "#computetemplateconfig-legacy"
)


def check_compute_configs_legacy(entries: list[Entry]) -> list[str]:
    """Each compute config file must follow the legacy ComputeTemplateConfig
    schema. Top-level keys are checked (nested keys are not validated)."""
    errors: list[str] = []
    paths: set[str] = set()
    for e in entries:
        paths.add(e.compute_config.GCP)
        paths.add(e.compute_config.AWS)
    for path in sorted(paths):
        full = REPO_ROOT / path
        if not full.is_file():
            continue  # already reported by check_filesystem_and_uniqueness
        try:
            data = yaml.safe_load(full.read_text()) or {}
            LegacyComputeTemplateConfig.model_validate(data)
        except ValidationError as e:
            extras = [
                ".".join(str(p) for p in err["loc"]) for err in e.errors()
                if err.get("type") == "extra_forbidden"
            ]
            if extras:
                errors.append(
                    f"{path}: unknown keys {extras} — this config likely uses "
                    f"the new compute-config schema. This repo requires the "
                    f"legacy ComputeTemplateConfig API: {LEGACY_DOCS}"
                )
            else:
                errors.append(f"{path}: {e.errors()}")
    return errors


# ----------------------------------------------- GCP image naming + access

GCP_BYOD_REGISTRY = (
    "us-docker.pkg.dev/anyscale-workspace-templates/workspace-templates"
)


def is_gcp_uri(uri: str) -> bool:
    host = uri.split("/", 1)[0]
    return host.endswith(".pkg.dev") or host.endswith("gcr.io")


def parse_ref(uri: str):
    repo, _, tag = uri.partition(":")
    if not tag:
        tag = "latest"
    if "/" not in repo:
        host, repo_path = "registry-1.docker.io", repo
    else:
        host, repo_path = repo.split("/", 1)
    return host, repo_path, tag


def manifest_accessible(uri: str) -> tuple[bool, str]:
    host, repo, tag = parse_ref(uri)
    url = f"https://{host}/v2/{repo}/manifests/{tag}"
    req = urllib.request.Request(url, method="HEAD")
    req.add_header("Accept", ACCEPT)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200, "ok"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return False, f"network error: {e.reason}"


def check_gcp_byod_images(entries: list[Entry], *, network: bool) -> list[str]:
    """Walk byod images hosted on a GCP registry: enforce the canonical
    `<registry>/<name>:<byod.ray_version>` form for the Anyscale registry, and
    (when `network` is on) verify the manifest is reachable. `image_uri` is
    skipped — the schema constrains it to `^anyscale/`, never a GCP host."""
    errors: list[str] = []
    for e in entries:
        if not e.cluster_env.byod:
            continue
        img = e.cluster_env.byod.docker_image
        if not is_gcp_uri(img):
            continue

        if img.startswith(f"{GCP_BYOD_REGISTRY}/"):
            expected = f"{GCP_BYOD_REGISTRY}/{e.name}:{e.cluster_env.byod.ray_version}"
            if img != expected:
                errors.append(
                    f"{e.name}.cluster_env.byod.docker_image: must be "
                    f"{expected!r}, got {img!r} (image basename must equal "
                    f"`name` and tag must equal `byod.ray_version`)"
                )

        if network:
            ok, msg = manifest_accessible(img)
            if not ok:
                errors.append(
                    f"{e.name}: GCP image not accessible: {img} ({msg}) — "
                    f"the registry is public, so this almost always means the "
                    f"push did not complete; rebuild and republish"
                )
    return errors


# ------------------------------------------------------------------ main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Skip the GCP image accessibility check (for pre-commit / offline use)",
    )
    args = parser.parse_args()

    raw = yaml.safe_load((REPO_ROOT / "BUILD.yaml").read_text())

    try:
        entries = TypeAdapter(list[Entry]).validate_python(raw)
    except ValidationError as e:
        print("FAIL: BUILD.yaml schema errors:", file=sys.stderr)
        for err in e.errors():
            loc = ".".join(str(p) for p in err["loc"])
            print(f"::error::{loc}: {err['msg']}", file=sys.stderr)
        return 1

    for w in check_redundant_compute_configs(entries):
        print(f"::warning::{w}", file=sys.stderr)

    errors = check_filesystem_and_uniqueness(entries)
    errors.extend(check_compute_configs_legacy(entries))
    errors.extend(check_gcp_byod_images(entries, network=not args.no_network))

    if errors:
        print(f"FAIL: {len(errors)} validation error(s):", file=sys.stderr)
        for err in errors:
            print(f"::error::{err}", file=sys.stderr)
        return 1

    mode = "schema + paths" if args.no_network else "schema + paths + GCP images"
    print(f"OK: validated {len(entries)} BUILD.yaml entries ({mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
