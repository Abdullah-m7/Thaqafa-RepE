"""The README's summary table must agree with the committed artefacts.

The table is written by hand, which is how it fell out of step once already:
a seed fix changed the gpt2 layer sweep and the README kept quoting the old
figures. Numbers a reader meets first are the ones most worth pinning, so each
cell is recomputed here from the CSVs under ``results/`` and compared.

If a rerun legitimately changes a number, this test fails and the README is
updated in the same commit as the artefact - which is the point.
"""

from __future__ import annotations

import csv
import re
import subprocess
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = PROJECT_ROOT / "README.md"
RESULTS = PROJECT_ROOT / "results"

MIN_READABLE_PROBE = 0.70
"""Probe quality below which a causal point is not counted, per the README."""


def _rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV into a list of row dicts."""
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _best_layer_rows(run: Path) -> list[dict[str, str]]:
    """Return each concept's highest-scoring layer from the sweep."""
    best: dict[str, dict[str, str]] = {}
    for row in _rows(run / "layer_sweep.csv"):
        concept = row["concept_id"]
        if concept not in best or float(row["probe_score"]) > float(best[concept]["probe_score"]):
            best[concept] = row
    return list(best.values())


def _causal_counts(path: Path, effect_key: str, probe_key: str) -> tuple[int, int]:
    """Count points through a usable probe, and how many show the effect."""
    readable = [row for row in _rows(path) if float(row[probe_key]) >= MIN_READABLE_PROBE]
    return sum(1 for row in readable if float(row[effect_key]) > 0), len(readable)


def _readme_row(label: str) -> list[str]:
    """Return the README summary-table cells for one run.

    Args:
        label: Text identifying the run's row, e.g. ``"pilot_gpt2"``.

    Returns:
        The row's cells, stripped.

    Raises:
        AssertionError: If the row is not in the table.
    """
    for line in README.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and f"results/{label}/" in line:
            return [cell.strip() for cell in line.strip("|").split("|")]
    raise AssertionError(f"No README summary row for {label}")


def _committed_runs() -> list[str]:
    """Return the run directories git is tracking, not whatever is on disk.

    The README summarises committed artefacts. Reading the filesystem instead
    would make this suite fail for the whole duration of any pipeline run,
    since a run creates its output directory before it has anything to report
    - which would train everyone to ignore a red suite.

    Returns:
        Sorted directory names under ``results/`` that contain tracked files.
    """
    listing = subprocess.run(  # noqa: S603
        ["git", "ls-files", "results"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if listing.returncode != 0:  # pragma: no cover - not a git checkout
        return []
    return sorted({line.split("/")[1] for line in listing.stdout.split() if "/" in line})


def _pilot_runs(runs: list[str]) -> list[str]:
    """Narrow to the runs the summary table actually describes.

    ``results/`` holds more than one kind of experiment. A pilot run produces
    a layer sweep and the causal CSVs, and has a row in the README's table; a
    standalone experiment such as the pair-tightness sweep produces neither and
    is written up in prose instead. Keying off the directory alone treated the
    second kind as a broken pilot, which is how this test failed the first time
    a different experiment was committed.

    Args:
        runs: Every committed run directory.

    Returns:
        Those carrying a ``layer_sweep.csv``.
    """
    return [run for run in runs if (RESULTS / run / "layer_sweep.csv").exists()]


RUNS = _committed_runs()
PILOT_RUNS = _pilot_runs(RUNS)


@pytest.mark.parametrize("run", PILOT_RUNS)
class TestReadmeMatchesArtefacts:
    """Every cell in the summary table, against the CSV behind it."""

    def test_significant_concept_count(self, run: str) -> None:
        """The headline "readable above chance" fraction."""
        best = _best_layer_rows(RESULTS / run)
        significant = sum(1 for row in best if float(row["p_value"]) < 0.05)

        assert _readme_row(run)[1] == f"{significant} / {len(best)}"

    def test_mean_balanced_accuracy(self, run: str) -> None:
        """Quoted to three decimals, so it must round to the same string."""
        best = _best_layer_rows(RESULTS / run)
        mean = sum(float(row["probe_score"]) for row in best) / len(best)

        assert _readme_row(run)[2] == _fmt3(mean)

    def test_amplification_counts(self, run: str) -> None:
        """Points where adding the direction beat a matched-norm random one."""
        hits, total = _causal_counts(
            RESULTS / run / "causal_readback.csv", "lift_over_random", "probe_accuracy"
        )

        assert _readme_row(run)[3] == f"{hits} / {total}"

    def test_suppression_counts(self, run: str) -> None:
        """Points where subtracting it beat a matched-norm random one."""
        hits, total = _causal_counts(
            RESULTS / run / "suppression.csv",
            "drop_beyond_random",
            "probe_balanced_accuracy",
        )

        assert _readme_row(run)[4] == f"{hits} / {total}"


def test_every_committed_run_appears_in_the_readme() -> None:
    """A run nobody links to is a run nobody reads."""
    text = README.read_text(encoding="utf-8")
    missing = [run for run in RUNS if f"results/{run}/" not in text]
    assert not missing, f"Committed but unlinked from the README: {missing}"


def test_at_least_one_run_is_committed() -> None:
    """Guards the parametrisation: an empty RUNS would silently pass everything."""
    assert RUNS
    assert PILOT_RUNS


def test_every_experiment_that_is_not_a_pilot_is_still_written_up() -> None:
    """A standalone experiment has no table row, so prose is what covers it.

    Without this, narrowing the table's parametrisation would have quietly
    dropped such a run from every check the README makes.
    """
    text = README.read_text(encoding="utf-8")
    for run in set(RUNS) - set(PILOT_RUNS):
        assert f"results/{run}/" in text, f"{run} is committed but not described"


def test_the_readme_states_the_probe_threshold_it_counts_by() -> None:
    """The causal fractions are meaningless without it."""
    text = README.read_text(encoding="utf-8")
    assert re.search(r"0\.70 balanced accuracy", text)


TIGHTNESS_RUN = RESULTS / "pair_tightness_qwen"
"""The pair-tightness experiment, whose two tables are the repository's
most consequential claim and so the ones least safe to leave unpinned."""

CONDITION_LABELS = {
    "neutral": "Generic neutral bank",
    "shipped": "The dataset's own contrasts",
    "tight": "Rewritten to be minimal",
}
"""README row label for each condition in ``tightness.csv``."""


def _fmt3(value: float) -> str:
    """Format to three decimals, rounding halves up.

    The tight condition's mean is exactly 0.5625, so ``format`` would return
    "0.562" or "0.563" depending on which way the float accumulation landed -
    a table cell that changes with summation order is not a pinned number.
    """
    return str(Decimal(repr(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _labelled_row(label: str) -> list[str]:
    """Return the cells of the README table row named by ``label``.

    A row is named by ``label`` when its first cell is exactly that, or is
    that followed by a space and a gloss - ``valence`` names the row
    "`valence` (delighted / disappointed)". Requiring the space is what keeps
    ``valence`` from also naming the ``valence_large`` row, whose two figures
    carry the opposite half of the argument.

    Args:
        label: The row's name, without markdown emphasis or code markers.

    Returns:
        The row's cells, stripped of whitespace and markdown emphasis.

    Raises:
        AssertionError: If no row, or more than one row, is so named.
    """
    matches = []
    for line in README.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [
            cell.strip().replace("*", "").replace("`", "") for cell in line.strip("|").split("|")
        ]
        if cells[0] == label or cells[0].startswith(f"{label} "):
            matches.append(cells)
    assert matches, f"No README table row named {label!r}"
    assert len(matches) == 1, f"{label!r} names {len(matches)} README rows, so it names none"
    return matches[0]


@pytest.mark.parametrize("condition", sorted(CONDITION_LABELS))
class TestTightnessTable:
    """Each contrast-set row, against ``tightness.csv``."""

    @staticmethod
    def _rows_for(condition: str) -> list[dict[str, str]]:
        rows = [r for r in _rows(TIGHTNESS_RUN / "tightness.csv") if r["condition"] == condition]
        assert rows, f"No {condition!r} rows in tightness.csv"
        return rows

    def test_vocabulary_overlap(self, condition: str) -> None:
        """The overlap that makes the monotone fall interpretable."""
        rows = self._rows_for(condition)
        mean = sum(float(r["vocabulary_overlap"]) for r in rows) / len(rows)

        assert _labelled_row(CONDITION_LABELS[condition])[1] == f"{round(mean * 100)}%"

    def test_mean_balanced_accuracy(self, condition: str) -> None:
        """The fall itself."""
        rows = self._rows_for(condition)
        mean = sum(float(r["probe_score"]) for r in rows) / len(rows)

        assert _labelled_row(CONDITION_LABELS[condition])[2] == _fmt3(mean)

    def test_significant_count(self, condition: str) -> None:
        """How many concepts survive a permutation test in this condition."""
        rows = self._rows_for(condition)
        significant = sum(1 for r in rows if float(r["p_value"]) < 0.05)

        assert _labelled_row(CONDITION_LABELS[condition])[3] == f"{significant} / {len(rows)}"


def _control_ids() -> list[str]:
    """Return the positive controls actually present in the committed CSV."""
    return [row["concept_id"] for row in _rows(TIGHTNESS_RUN / "positive_controls.csv")]


@pytest.mark.parametrize("control", _control_ids())
class TestPositiveControlTable:
    """Each control row, against ``positive_controls.csv``.

    These are what separate "the concepts are not represented" from "this
    design cannot see a one-word difference at twelve samples", so a drifted
    cell here would misstate the repository's conclusion.
    """

    @staticmethod
    def _row_for(control: str) -> dict[str, str]:
        for row in _rows(TIGHTNESS_RUN / "positive_controls.csv"):
            if row["concept_id"] == control:
                return row
        raise AssertionError(f"No {control!r} row in positive_controls.csv")

    def test_vocabulary_overlap(self, control: str) -> None:
        row = self._row_for(control)

        assert _labelled_row(control)[1] == f"{round(float(row['vocabulary_overlap']) * 100)}%"

    def test_sample_count(self, control: str) -> None:
        """The column the power argument turns on: 12 versus 24."""
        row = self._row_for(control)
        samples = int(row["n_positive"]) + int(row["n_negative"])

        assert _labelled_row(control)[2] == str(samples)

    def test_balanced_accuracy(self, control: str) -> None:
        row = self._row_for(control)

        assert _labelled_row(control)[3] == _fmt3(float(row["probe_score"]))

    def test_p_value(self, control: str) -> None:
        row = self._row_for(control)

        assert _labelled_row(control)[4] == _fmt3(float(row["p_value"]))


def test_the_readme_reports_every_committed_positive_control() -> None:
    """A control run but not shown is a control the reader cannot weigh."""
    assert _control_ids(), "positive_controls.csv is empty"
    for control in _control_ids():
        _labelled_row(control)
