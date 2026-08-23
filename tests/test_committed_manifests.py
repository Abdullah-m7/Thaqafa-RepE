"""Every committed manifest must still describe the files it was run against.

``tests/test_provenance.py`` covers the code that *writes* a manifest. Nothing
covered the manifests already sitting in ``results/``, so an input could be
edited after the fact and the run that quotes its hash would go on claiming
provenance it no longer has - the one failure mode recording hashes exists to
catch.

This reads only tracked files, so a scratch run on disk cannot make it pass or
fail.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HASH_SETTINGS = {
    "tight_contrasts_sha256": "data/experiments/tight_contrasts_en.jsonl",
    "positive_controls_sha256": "data/experiments/positive_controls_en.jsonl",
}
"""Settings keys that pin an input file, mapped to the file they pin."""


def _tracked_manifests() -> list[str]:
    """Return the manifests git is tracking, newest layout first."""
    listed = subprocess.run(
        ["git", "ls-files", "results/*/manifest.json"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(listed.stdout.split())


def _is_shallow() -> bool:
    """Whether this checkout was cloned without full history."""
    answer = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return answer.stdout.strip() == "true"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


MANIFESTS = _tracked_manifests()


def test_at_least_one_manifest_is_committed() -> None:
    """Guards the parametrisation: an empty list would pass everything below."""
    assert MANIFESTS


@pytest.mark.parametrize("manifest_path", MANIFESTS)
class TestCommittedManifest:
    """Each committed run, against the inputs it names."""

    @staticmethod
    def _load(manifest_path: str) -> dict:
        return json.loads((PROJECT_ROOT / manifest_path).read_text(encoding="utf-8"))

    def test_dataset_hash_still_matches(self, manifest_path: str) -> None:
        """The dataset has not been edited since this run quoted it."""
        manifest = self._load(manifest_path)
        dataset = PROJECT_ROOT / manifest["dataset"]["path"]

        assert dataset.exists(), f"{manifest_path} names a dataset that is gone"
        assert _digest(dataset) == manifest["dataset"]["sha256"], (
            f"{manifest_path} was run against a different "
            f"{manifest['dataset']['path']} than the one committed"
        )

    def test_pinned_input_hashes_still_match(self, manifest_path: str) -> None:
        """Same, for the experiment-specific inputs a run may pin."""
        settings = self._load(manifest_path).get("settings", {})
        for key, relative in HASH_SETTINGS.items():
            if key not in settings:
                continue
            path = PROJECT_ROOT / relative
            assert path.exists(), f"{manifest_path} pins {relative}, which is gone"
            assert _digest(path) == settings[key], (
                f"{manifest_path} was run against a different {relative} " "than the one committed"
            )

    def test_the_run_came_from_a_clean_tree(self, manifest_path: str) -> None:
        """A dirty tree means the committed code is not the code that ran.

        Changes under ``results/`` are counted separately by the provenance
        helper, precisely so a run writing its own output does not flag itself.
        """
        git = self._load(manifest_path)["git"]

        assert git["dirty"] is False, f"{manifest_path} records an uncommitted working tree"
        assert git["untracked"] == 0, f"{manifest_path} records untracked files"

    def test_the_recorded_commit_is_a_full_sha(self, manifest_path: str) -> None:
        """An abbreviated or truncated sha stops identifying a commit as history grows."""
        commit = self._load(manifest_path)["git"]["commit"]

        assert re.fullmatch(
            r"[0-9a-f]{40}", commit
        ), f"{manifest_path} records {commit!r}, which is not a full commit sha"

    def test_the_recorded_commit_exists(self, manifest_path: str) -> None:
        """A commit nobody can check out is not provenance.

        Skipped on a shallow clone, which is what ``actions/checkout`` makes by
        default: there, an older commit is missing because the history was never
        fetched, so its absence says nothing about the manifest. Asserting it
        anyway failed CI on four manifests that were perfectly valid.
        """
        if _is_shallow():
            pytest.skip("shallow clone: absent history is not a missing commit")

        commit = self._load(manifest_path)["git"]["commit"]
        found = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=PROJECT_ROOT,
            capture_output=True,
        )

        assert found.returncode == 0, f"{manifest_path} names unknown commit {commit}"
