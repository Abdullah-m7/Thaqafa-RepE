#!/usr/bin/env python3
"""Check the concept dataset, and flag pairs that are not actually minimal.

The test suite already enforces the invariants a dataset must never violate.
This is the other half: things that are not errors but are worth a human
looking at, reported as warnings a reviewer can act on or dismiss.

The one that matters most is the minimal pair. Extraction subtracts the mean
of the contrasts from the mean of the exemplars, so whatever the two sets share
cancels and whatever differs survives. If a contrast changes the subject, the
surviving direction is about the subject. If a contrast still carries the
concept, there is no direction left to find. Neither shows up as a broken
field, and neither is visible from inside a single sentence - only from the
overlap between the two sets.

Exit status is 0 unless ``--strict`` is passed, because a warning here is an
invitation to look, not a verdict.

Usage:
    python scripts/check_dataset.py
    python scripts/check_dataset.py --strict
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_builder import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    CulturalConcept,
    load_concepts,
    review_summary,
)

ARABIC_RANGE = re.compile(r"[؀-ۿ]+")
LATIN_RANGE = re.compile(r"[A-Za-z]+")

MIN_SHARED_FRACTION = 0.25
"""Below this, exemplars and contrasts look like they are about different things."""

RARE_WORD_MAX_USES = 1
"""A content word this rare on one side is the kind of thing a pair should cancel."""

FUNCTION_WORDS = frozenset(
    {"and", "the", "for", "with", "from", "into", "that", "this", "his", "her", "its"}
)
"""Excluded when matching a concept's *name* against the contrast side.

A gloss like "Modesty and Propriety" contributes "and", which then matches
almost any sentence and reports a leak that is not one. Only the content words
of a name can indicate that the concept itself appears on both sides.
"""


def _words(sentences: list[str]) -> Counter[str]:
    """Count word occurrences across sentences, in either script."""
    counts: Counter[str] = Counter()
    for sentence in sentences:
        for match in ARABIC_RANGE.findall(sentence) + LATIN_RANGE.findall(sentence):
            if len(match) > 2:
                counts[match.lower()] += 1
    return counts


def check_minimal_pairs(concept: CulturalConcept, language: str) -> list[str]:
    """Report ways one language's exemplar/contrast pair fails to be minimal.

    Args:
        concept: The entry to inspect.
        language: ``"ar"`` or ``"en"``.

    Returns:
        Human-readable warnings, empty when nothing looks wrong.
    """
    exemplars = concept.examples_ar if language == "ar" else concept.examples_en
    contrasts = concept.contrast_ar if language == "ar" else concept.contrast_en
    if not exemplars or not contrasts:
        return [f"[{language}] no {'exemplars' if not exemplars else 'contrasts'} to compare"]

    exemplar_words = _words(exemplars)
    contrast_words = _words(contrasts)
    shared = set(exemplar_words) & set(contrast_words)
    union = set(exemplar_words) | set(contrast_words)

    warnings = []
    fraction = len(shared) / len(union) if union else 0.0
    if fraction < MIN_SHARED_FRACTION:
        warnings.append(
            f"[{language}] exemplars and contrasts share only {fraction:.0%} of their "
            f"vocabulary. A pair that changes the subject cancels nothing, so the "
            f"extracted direction may be about the topic rather than the concept."
        )

    # A word carrying the concept should not appear on the contrast side at all.
    concept_words = set(_words([concept.concept_ar, concept.concept_en])) - FUNCTION_WORDS
    leaked = sorted(word for word in concept_words if word in contrast_words)
    if leaked:
        warnings.append(
            f"[{language}] the contrasts contain the concept's own name "
            f"({', '.join(leaked)}). If the concept is present on both sides "
            f"there is no direction left to extract."
        )

    only_contrast = sorted(
        word
        for word, count in contrast_words.items()
        if word not in exemplar_words and count > RARE_WORD_MAX_USES
    )
    if only_contrast:
        warnings.append(
            f"[{language}] repeated on the contrast side and never on the exemplar "
            f"side: {', '.join(only_contrast[:6])}. Whatever is unique to one side "
            f"survives the subtraction."
        )
    return warnings


def pair_overlap(concept: CulturalConcept, language: str) -> float:
    """Return the vocabulary overlap between exemplars and contrasts.

    The single number that says how minimal a "minimal pair" actually is.
    Extraction subtracts one mean from the other, so shared vocabulary cancels
    and unshared vocabulary survives: a pair sharing almost nothing yields a
    direction that is mostly the difference in subject matter.

    Args:
        concept: The entry to measure.
        language: ``"ar"`` or ``"en"``.

    Returns:
        Shared words as a fraction of all words used by either side, or 0.0
        when a side is empty.
    """
    exemplars = concept.examples_ar if language == "ar" else concept.examples_en
    contrasts = concept.contrast_ar if language == "ar" else concept.contrast_en
    if not exemplars or not contrasts:
        return 0.0
    exemplar_words = set(_words(exemplars))
    contrast_words = set(_words(contrasts))
    union = exemplar_words | contrast_words
    return len(exemplar_words & contrast_words) / len(union) if union else 0.0


def check_concept(concept: CulturalConcept) -> list[str]:
    """Collect every warning for one concept.

    Args:
        concept: The entry to inspect.

    Returns:
        Human-readable warnings.
    """
    warnings = []
    for language in ("ar", "en"):
        warnings.extend(check_minimal_pairs(concept, language))

    if len(concept.examples_ar) != len(concept.examples_en):
        warnings.append(
            f"{len(concept.examples_ar)} Arabic exemplars against "
            f"{len(concept.examples_en)} English ones. The cross-lingual checks "
            f"compare directions built from each side, so an uneven split makes "
            f"one of them noisier for a reason unrelated to language."
        )
    if not concept.cultural_context.strip():
        warnings.append("no cultural_context: a reader has nothing to judge the framing against")
    return warnings


def main(argv: list[str] | None = None) -> int:
    """Report warnings for the dataset.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        ``1`` under ``--strict`` when anything was flagged, else ``0``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if anything is flagged. Off by default: a warning "
        "here is an invitation to look, not a verdict.",
    )
    args = parser.parse_args(argv)

    concepts = load_concepts(Path(args.dataset))
    counts = review_summary(concepts)

    print(f"{'concept':18} {'ar overlap':>11} {'en overlap':>11}")
    print("-" * 42)
    for concept in sorted(concepts, key=lambda c: -pair_overlap(c, "ar")):
        print(
            f"{concept.concept_id:18} {pair_overlap(concept, 'ar'):10.0%} "
            f"{pair_overlap(concept, 'en'):10.0%}"
        )
    print(
        "\nOverlap is how much vocabulary an entry's exemplars and contrasts\n"
        "share. Higher is tighter. A pair sharing almost nothing is not a\n"
        "minimal pair: whatever is unique to one side survives the subtraction,\n"
        "so the extracted direction carries the difference in subject matter\n"
        "along with - or instead of - the concept."
    )

    total = 0
    for concept in concepts:
        warnings = check_concept(concept)
        if not warnings:
            continue
        total += len(warnings)
        print(f"\n{concept.concept_id} ({concept.concept_ar})")
        for warning in warnings:
            print(f"  - {warning}")

    print(
        f"\n{counts['reviewed']}/{counts['total']} entries carry a named native-speaker "
        f"review. {total} thing(s) flagged for a human to look at."
    )
    return 1 if (args.strict and total) else 0


if __name__ == "__main__":
    raise SystemExit(main())
