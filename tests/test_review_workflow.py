"""Tests for the review sheet, the dataset checker, and review attribution.

Native-speaker review is the one step in this pipeline a machine cannot do, so
the tooling around it has two jobs: make the judgement cheap, and make the
resulting claim checkable. These tests are about the second - that "reviewed"
cannot be asserted without evidence, and that the checker actually notices the
failure the method is most vulnerable to.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_review_sheet, check_dataset  # noqa: E402
from src.data.dataset_builder import CulturalConcept, load_concepts, review_summary  # noqa: E402
from tests.helpers import DATASET_PATH  # noqa: E402


def concept(**overrides: object) -> CulturalConcept:
    """Build a concept with sensible defaults for the field under test."""
    defaults: dict[str, object] = {
        "concept_id": "test_001",
        "concept_ar": "الكرم",
        "concept_en": "Karam",
        "category": "ethical",
        "description": "Generosity as a duty.",
        "examples_ar": ["أكرم ضيفه في بيته", "أكرم جاره في بيته"],
        "examples_en": ["He honoured his guest at home", "He honoured his neighbour at home"],
        "contrast_ar": ["استقبل ضيفه في بيته", "استقبل جاره في بيته"],
        "contrast_en": ["He received his guest at home", "He received his neighbour at home"],
        "cultural_context": "Widely held.",
    }
    defaults.update(overrides)
    return CulturalConcept(**defaults)  # type: ignore[arg-type]


class TestReviewAttribution:
    """ "Reviewed" is a claim, and a claim needs someone behind it."""

    def test_a_status_alone_does_not_count_as_reviewed(self) -> None:
        """Nobody could follow up, correct, or date-check such an entry."""
        assert concept(review_status="reviewed").is_reviewed is False

    def test_a_reviewer_without_a_date_does_not_count(self) -> None:
        """The exemplars change; an undated approval goes stale silently."""
        assert concept(review_status="reviewed", reviewed_by="A Name").is_reviewed is False

    def test_a_date_without_a_reviewer_does_not_count(self) -> None:
        assert concept(review_status="reviewed", reviewed_at="2026-01-01").is_reviewed is False

    def test_both_together_count(self) -> None:
        entry = concept(review_status="reviewed", reviewed_by="A Name", reviewed_at="2026-01-01")
        assert entry.is_reviewed is True

    def test_attribution_without_the_status_does_not_count(self) -> None:
        """Leftover attribution on a reverted entry must not read as approval."""
        entry = concept(reviewed_by="A Name", reviewed_at="2026-01-01")
        assert entry.is_reviewed is False

    def test_whitespace_is_not_a_reviewer(self) -> None:
        entry = concept(review_status="reviewed", reviewed_by="   ", reviewed_at="2026-01-01")
        assert entry.is_reviewed is False

    def test_the_fields_survive_a_round_trip_through_json(self) -> None:
        """A reviewer's note must not be dropped by the loader."""
        record: dict[str, object] = {
            "concept_id": "x_001",
            "concept_ar": "أ",
            "concept_en": "X",
            "category": "social",
            "description": "d",
            "review_status": "reviewed",
            "reviewed_by": "A Name",
            "reviewed_at": "2026-01-01",
            "review_notes": "kept sentence 3 under protest",
        }
        entry = CulturalConcept.from_dict(record)

        assert entry.reviewed_by == "A Name"
        assert entry.review_notes == "kept sentence 3 under protest"
        assert entry.is_reviewed is True


class TestReviewSummary:
    """Reports carry this instead of a hand-written sentence."""

    def test_counts_only_attributed_reviews(self) -> None:
        summary = review_summary(
            [
                concept(concept_id="a", review_status="reviewed"),
                concept(
                    concept_id="b",
                    review_status="reviewed",
                    reviewed_by="N",
                    reviewed_at="2026-01-01",
                ),
                concept(concept_id="c"),
            ]
        )
        assert summary == {"reviewed": 1, "total": 3}

    def test_an_empty_dataset_is_not_a_division_by_zero(self) -> None:
        assert review_summary([]) == {"reviewed": 0, "total": 0}

    def test_the_shipped_dataset_reports_honestly(self) -> None:
        """Whatever the number is, it must come from the file, not a comment."""
        summary = review_summary(load_concepts(DATASET_PATH))
        assert summary["total"] > 0
        assert 0 <= summary["reviewed"] <= summary["total"]


class TestPairOverlap:
    """The number that says how minimal a "minimal pair" really is."""

    def test_identical_wording_apart_from_the_concept_scores_high(self) -> None:
        """The ideal: same frame, same topic, one thing changed."""
        assert check_dataset.pair_overlap(concept(), "ar") > 0.4

    def test_changing_the_subject_scores_low(self) -> None:
        """Whatever is unique to one side survives the subtraction."""
        loose = concept(
            contrast_ar=["ذهب الطالب إلى الجامعة صباحا", "قرأ الطالب كتابا طويلا"],
        )
        assert check_dataset.pair_overlap(loose, "ar") < 0.2

    def test_an_empty_side_is_zero_not_an_error(self) -> None:
        assert check_dataset.pair_overlap(concept(contrast_ar=[]), "ar") == 0.0

    def test_it_is_reported_per_language(self) -> None:
        entry = concept()
        assert check_dataset.pair_overlap(entry, "ar") > 0
        assert check_dataset.pair_overlap(entry, "en") > 0


class TestChecker:
    """Warnings a reviewer can act on."""

    def test_flags_a_pair_that_changed_the_subject(self) -> None:
        loose = concept(contrast_ar=["ذهب الطالب إلى الجامعة", "قرأ الطالب كتابا"])
        warnings = check_dataset.check_minimal_pairs(loose, "ar")
        assert any("vocabulary" in warning for warning in warnings)

    def test_flags_a_contrast_that_still_carries_the_concept(self) -> None:
        """If the concept is on both sides there is no direction to extract."""
        leaky = concept(contrast_ar=["أظهر الكرم مع ضيفه في بيته", "الكرم في بيته"])
        warnings = check_dataset.check_minimal_pairs(leaky, "ar")
        assert any("own name" in warning for warning in warnings)

    def test_a_function_word_in_a_gloss_is_not_a_leak(self) -> None:
        """ "Modesty and Propriety" contributes "and", which matches anything.

        Only the content words of a concept's name can show that the concept
        itself sits on both sides of the pair.
        """
        entry = concept(
            concept_en="Modesty and Propriety",
            contrast_ar=["استقبل ضيفه وجاره في بيته"],
        )
        warnings = check_dataset.check_minimal_pairs(entry, "en")
        assert not any("own name" in warning for warning in warnings)

    def test_flags_an_uneven_bilingual_split(self) -> None:
        uneven = concept(examples_en=["only one"])
        assert any("against" in warning for warning in check_dataset.check_concept(uneven))

    def test_a_tight_pair_raises_no_vocabulary_warning(self) -> None:
        warnings = check_dataset.check_minimal_pairs(concept(), "ar")
        assert not any("vocabulary" in warning for warning in warnings)

    def test_missing_context_is_flagged(self) -> None:
        assert any(
            "cultural_context" in warning
            for warning in check_dataset.check_concept(concept(cultural_context=""))
        )

    def test_strict_mode_fails_when_anything_is_flagged(self, tmp_path: Path) -> None:
        """So the check can gate a pull request once the dataset is clean."""
        assert check_dataset.main(["--dataset", str(DATASET_PATH), "--strict"]) == 1

    def test_default_mode_never_fails(self) -> None:
        """A warning is an invitation to look, not a verdict."""
        assert check_dataset.main(["--dataset", str(DATASET_PATH)]) == 0


class TestReviewSheet:
    """The document that makes the judgement cheap."""

    def test_puts_the_arabic_first(self) -> None:
        """The Arabic is what is being judged; the English is reference."""
        text = "\n".join(build_review_sheet.concept_section(concept()))
        assert text.index("Arabic exemplars") < text.index("English side")

    def test_says_which_file_and_fields_to_edit(self) -> None:
        sheet = build_review_sheet.build_sheet([concept()], DATASET_PATH)
        assert "reviewed_by" in sheet
        assert "reviewed_at" in sheet
        assert DATASET_PATH.name in sheet

    def test_says_that_changing_a_sentence_is_a_normal_outcome(self) -> None:
        """A reviewer who thinks only approval is wanted will approve."""
        sheet = build_review_sheet.build_sheet([concept()], DATASET_PATH)
        assert "not a failure" in sheet

    def test_explains_what_a_contrast_is_for(self) -> None:
        """It is the part of the method a reviewer would not otherwise guess."""
        text = "\n".join(build_review_sheet.concept_section(concept()))
        assert "absent" in text

    def test_marks_an_unreviewed_entry_as_awaiting_review(self) -> None:
        text = "\n".join(build_review_sheet.concept_section(concept()))
        assert "awaiting review" in text

    def test_an_empty_exemplar_list_is_itself_flagged(self) -> None:
        text = "\n".join(build_review_sheet.concept_section(concept(contrast_ar=[])))
        assert "worth flagging" in text

    def test_covers_every_concept_in_the_shipped_dataset(self, tmp_path: Path) -> None:
        output = tmp_path / "sheet.md"
        build_review_sheet.main(["--output", str(output), "--dataset", str(DATASET_PATH)])
        text = output.read_text(encoding="utf-8")

        for entry in load_concepts(DATASET_PATH):
            assert entry.concept_id in text

    def test_pending_only_skips_attributed_entries(self, tmp_path: Path) -> None:
        """So a second review pass is not handed work already done."""
        output = tmp_path / "sheet.md"
        build_review_sheet.main(
            ["--output", str(output), "--dataset", str(DATASET_PATH), "--pending-only"]
        )
        reviewed = [c.concept_id for c in load_concepts(DATASET_PATH) if c.is_reviewed]
        text = output.read_text(encoding="utf-8")

        for concept_id in reviewed:
            assert concept_id not in text

    def test_an_unknown_concept_id_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="Unknown concept"):
            build_review_sheet.main(
                ["--output", str(tmp_path / "s.md"), "--concepts", "not_a_concept_999"]
            )
