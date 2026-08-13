#!/usr/bin/env python3
"""Unit tests for prepare-base-locks.py — the minor-only version gate.

Hermetic: mocks the two version-resolution funcs (newest_stable_ray = PyPI,
newest_complete = local depsets), so nothing touches the network or the real
dependencies/depsets/. The prepare path (which loads the config via ruamel) is
stubbed with _yaml, so these tests need neither ruamel nor the real config —
matching CI's pre-commit job, which installs requirements-dev.txt (no ruamel).

Run directly: python3 scripts/ray-bump/test_prepare_base_locks.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# The script's filename has hyphens, so load it by path rather than import.
_MODULE_PATH = Path(__file__).with_name("prepare-base-locks.py")
_spec = importlib.util.spec_from_file_location("prepare_base_locks", _MODULE_PATH)
pbl = importlib.util.module_from_spec(_spec)
sys.path.insert(0, str(Path(__file__).parent))
_spec.loader.exec_module(pbl)

import depset_versions as dv  # noqa: E402  (needs the sys.path insert above)


class _ReachedPrepare(Exception):
    """Stands in for _yaml() to prove main() got past the gate into the prepare
    body without needing ruamel or the real config."""


def _run_main(argv, **returns):
    """Run pbl.main(argv) with the named funcs patched to return `returns[name]`,
    _yaml stubbed to raise _ReachedPrepare, and GITHUB_OUTPUT captured.

    Returns (rc, reached, outputs): rc is main()'s exit code (None if it reached
    the prepare body), reached is True iff it got there, outputs is the parsed
    GITHUB_OUTPUT dict.
    """
    with tempfile.NamedTemporaryFile("w+", suffix=".out") as out, contextlib.ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"GITHUB_OUTPUT": out.name}))
        for name, val in returns.items():
            stack.enter_context(patch.object(pbl, name, Mock(return_value=val)))
        stack.enter_context(patch.object(pbl, "_yaml", Mock(side_effect=_ReachedPrepare)))
        try:
            rc, reached = pbl.main(argv), False
        except _ReachedPrepare:
            rc, reached = None, True
        out.seek(0)
        outputs = dict(ln.split("=", 1) for ln in out.read().splitlines() if "=" in ln)
    return rc, reached, outputs


class MinorUpgradeTest(unittest.TestCase):
    """The pure gate decision."""

    def test_patch_over_current_minor_is_not_an_upgrade(self):
        self.assertFalse(pbl.is_minor_upgrade("2.56.1", "2.56.0"))

    def test_new_minor_is_an_upgrade(self):
        self.assertTrue(pbl.is_minor_upgrade("2.57.0", "2.56.0"))
        self.assertTrue(pbl.is_minor_upgrade("2.57.2", "2.56.0"))

    def test_new_major_is_an_upgrade(self):
        self.assertTrue(pbl.is_minor_upgrade("3.0.0", "2.56.0"))

    def test_same_version_is_not_an_upgrade(self):
        self.assertFalse(pbl.is_minor_upgrade("2.56.0", "2.56.0"))

    def test_behind_current_is_not_an_upgrade(self):
        self.assertFalse(pbl.is_minor_upgrade("2.55.9", "2.56.0"))

    def test_bootstrap_no_current_is_an_upgrade(self):
        self.assertTrue(pbl.is_minor_upgrade("2.56.0", None))


class MainGateTest(unittest.TestCase):
    """The gate as wired into main() (scheduled/auto path vs --version bypass)."""

    def test_auto_patch_bump_is_a_noop(self):
        # current 2.56.0, PyPI newest 2.56.1 (patch) -> skip, rc=0.
        rc, reached, out = _run_main([], newest_stable_ray="2.56.1", newest_complete="2.56.0")
        self.assertEqual(rc, 0)
        self.assertFalse(reached)
        self.assertEqual(out.get("status"), "skipped-patch")

    def test_auto_already_current_is_a_noop(self):
        # current 2.56.0, PyPI newest 2.56.0 -> not ahead, rc=0.
        rc, reached, _ = _run_main([], newest_stable_ray="2.56.0", newest_complete="2.56.0")
        self.assertEqual(rc, 0)
        self.assertFalse(reached)

    def test_auto_new_minor_proceeds(self):
        # current 2.56.0, PyPI newest 2.57.0 -> proceeds into the prepare body.
        _, reached, _ = _run_main(
            [], newest_stable_ray="2.57.0", newest_complete="2.56.0", complete_versions={"2.56.0"}
        )
        self.assertTrue(reached)

    def test_auto_new_minor_adopts_newest_patch(self):
        # current 2.56.0, PyPI newest 2.57.2 -> proceeds targeting 2.57.2 (not forced .0).
        _, reached, _ = _run_main(
            [], newest_stable_ray="2.57.2", newest_complete="2.56.0", complete_versions={"2.56.0"}
        )
        self.assertTrue(reached)

    def test_explicit_version_bypasses_gate_for_patch(self):
        # --version 2.56.1 (patch over 2.56.0) still proceeds — human override.
        _, reached, _ = _run_main(
            ["--version", "2.56.1"], newest_complete="2.56.0", complete_versions={"2.56.0"}
        )
        self.assertTrue(reached)

    def test_force_bypasses_gate_for_patch(self):
        # --force (no --version) also proceeds past the gate on a patch bump.
        _, reached, _ = _run_main(
            ["--force"], newest_stable_ray="2.56.1", newest_complete="2.56.0", complete_versions={"2.56.0"}
        )
        self.assertTrue(reached)


class NewestStableRayTest(unittest.TestCase):
    """newest_stable_ray picks the max final release (the newest minor's newest patch)."""

    def test_picks_newest_patch_of_newest_minor(self):
        payload = {
            "releases": {
                "2.56.0": [{"yanked": False}],
                "2.56.1": [{"yanked": False}],
                "2.57.0": [{"yanked": False}],
                "2.57.1": [{"yanked": False}],
                "2.57.2": [{"yanked": False}],
                "2.58.0rc1": [{"yanked": False}],  # non-final: ignored by the X.Y.Z regex
                "2.99.0": [{"yanked": True}],       # fully yanked: ignored
            }
        }

        class _Resp:
            def __enter__(self_):
                return self_

            def __exit__(self_, *a):
                return False

            def read(self_):
                return json.dumps(payload).encode()

        with patch.object(pbl.urllib.request, "urlopen", return_value=_Resp()):
            self.assertEqual(pbl.newest_stable_ray(), "2.57.2")


class TestCompleteness(unittest.TestCase):
    """The real freeze logic, against a fixture tree.

    The rest of this file mocks newest_complete out, which is how the probe's copy
    of this logic drifted from the resolver's unnoticed. These exercise it for real.
    """

    @contextlib.contextmanager
    def _tree(self, freezes):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "dependencies" / "depsets").mkdir(parents=True)
            images = root / "dependencies" / "images"
            images.mkdir(parents=True)
            (root / "dependencies" / "depsets" / "ray_2.99.0_img_py312.lock").touch()
            (images / "tracked-images.txt").write_text(
                "# comment\nanyscale/ray-llm:{version}-py312-cu130\n"
                "anyscale/ray:{version}-slim-py312-cu128\n"
            )
            for name, packages in freezes.items():
                (images / name).write_text("\n".join(f"pkg{i}==1.0" for i in range(packages)))
            with patch.object(dv, "DEPSETS", root / "dependencies" / "depsets"), \
                 patch.object(dv, "IMAGES", images):
                yield

    def test_all_freezes_present(self):
        with self._tree({"ray-llm-2.99.0-py312-cu130.freeze.txt": 300,
                         "ray-2.99.0-slim-py312-cu128.freeze.txt": 300}):
            self.assertEqual(dv.complete_versions(), {"2.99.0"})

    def test_partial_publish_is_incomplete(self):
        """One freeze present is not evidence the rest landed."""
        with self._tree({"ray-llm-2.99.0-py312-cu130.freeze.txt": 300}):
            self.assertEqual(dv.complete_versions(), set())

    def test_slim_freeze_counts(self):
        """A `slim-` freeze must be recognised — an earlier regex skipped them."""
        with self._tree({"ray-llm-2.99.0-py312-cu130.freeze.txt": 300,
                         "ray-2.99.0-slim-py312-cu128.freeze.txt": 300}):
            self.assertEqual(dv.missing_freezes("2.99.0"), [])

    def test_empty_freeze_counts_as_missing(self):
        """A failed fetch can leave a header-only file; it seeds nothing."""
        with self._tree({"ray-llm-2.99.0-py312-cu130.freeze.txt": 300,
                         "ray-2.99.0-slim-py312-cu128.freeze.txt": 0}):
            self.assertEqual(dv.complete_versions(), set())
            self.assertEqual([f.name for f in dv.missing_freezes("2.99.0")],
                             ["ray-2.99.0-slim-py312-cu128.freeze.txt"])

    def test_probe_and_resolver_agree(self):
        """The two callers must gate on the same definition."""
        self.assertIs(pbl.complete_versions, dv.complete_versions)


if __name__ == "__main__":
    unittest.main()
