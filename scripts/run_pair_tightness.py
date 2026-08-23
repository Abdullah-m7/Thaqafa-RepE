#!/usr/bin/env python3
"""Does a probe's score come from the concept, or from the contrast's topic?

The dataset's exemplars and contrasts share very little vocabulary, and across
the twelve concepts the tighter pairs score *worse* - a correlation of about
-0.6 between overlap and probe accuracy. That is suggestive and confounded:
the tighter-paired concepts might simply be harder for unrelated reasons.

This settles it by intervention rather than correlation. The exemplars are held
fixed and only the negative side changes, across three levels of tightness:

* **neutral** - the generic neutral bank, which shares nothing with the
  exemplars at all. The loosest contrast possible.
* **shipped** - the dataset's own curated contrasts.
* **tight** - contrasts written to reuse the exemplar's frame and change one
  thing, from ``data/experiments/tight_contrasts_en.jsonl``.

If accuracy falls as the pair tightens, the score was partly separating subject
matter, and the amount it falls is how much.

English only, deliberately. Writing a minimal pair is a question of
experimental design, and English is a language this repository's authors can
write minimal pairs in; the Arabic side is what native-speaker review is for,
and inventing more of it here would deepen the debt rather than measure it.

Usage:
    python scripts/run_pair_tightness.py --model Qwen/Qwen2.5-0.5B
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.contrastive import build_contrast_examples  # noqa: E402
from src.data.dataset_builder import DEFAULT_DATASET_PATH, load_concepts  # noqa: E402
from src.models.rep_engine import CulturalRepE  # noqa: E402
from src.utils.probes import DEFAULT_N_PERMUTATIONS, best_layer, probe_layer  # noqa: E402
from src.utils.provenance import (  # noqa: E402
    build_manifest,
    file_digest,
    set_global_seed,
)

logger = logging.getLogger("tightness")

TIGHT_CONTRASTS = PROJECT_ROOT / "data" / "experiments" / "tight_contrasts_en.jsonl"
DEFAULT_OUTPUT_DIR = "results/pair_tightness"
DEFAULT_SEED = 42
CONDITIONS = ("neutral", "shipped", "tight")
"""Contrast sets, ordered loosest to tightest."""


def load_tight_contrasts(path: Path = TIGHT_CONTRASTS) -> dict[str, list[str]]:
    """Read the tightened English contrasts.

    Args:
        path: JSONL file of ``{"concept_id": ..., "contrast_en": [...]}``.

    Returns:
        A mapping from concept id to its tightened contrasts.

    Raises:
        ValueError: If the file holds no entries.
    """
    contrasts: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        contrasts[str(record["concept_id"])] = [str(text) for text in record["contrast_en"]]
    if not contrasts:
        raise ValueError(f"No tightened contrasts in {path}")
    return contrasts


def vocabulary_overlap(exemplars: list[str], contrasts: list[str]) -> float:
    """Shared words as a fraction of all words either side uses.

    Args:
        exemplars: The positive side.
        contrasts: The negative side.

    Returns:
        A fraction in ``[0, 1]``.
    """
    from scripts.check_dataset import _words

    positive = set(_words(exemplars))
    negative = set(_words(contrasts))
    union = positive | negative
    return len(positive & negative) / len(union) if union else 0.0


def sweep_condition(
    engine: CulturalRepE,
    concept_id: str,
    exemplars: list[str],
    contrasts: list[str],
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Probe every layer for one concept under one contrast set.

    The permutation test runs only at the best layer: it costs far more than a
    single fit, and the layer sweep exists to find where to look, not to make
    twenty-four significance claims.

    Args:
        engine: Engine with a loaded model.
        concept_id: Concept being probed, for the row.
        exemplars: Positive prompts, identical across conditions.
        contrasts: Negative prompts for this condition.
        n_permutations: Shufflings behind the p-value at the best layer.
        seed: Seed for the probes and their folds.

    Returns:
        One row: the best layer, its score, its p-value and the pair's overlap.
    """
    n_layers = int(engine.model.cfg.n_layers)  # type: ignore[union-attr]
    results = {
        layer: probe_layer(
            engine,
            layer=layer,
            positive_prompts=exemplars,
            negative_prompts=contrasts,
            seed=seed,
            n_permutations=0,
        )
        for layer in range(n_layers)
    }
    layer = best_layer(results)
    confirmed = probe_layer(
        engine,
        layer=layer,
        positive_prompts=exemplars,
        negative_prompts=contrasts,
        seed=seed,
        n_permutations=n_permutations,
    )
    return {
        "concept_id": concept_id,
        "best_layer": layer,
        "probe_score": round(confirmed.accuracy, 6),
        "p_value": "" if confirmed.p_value is None else round(confirmed.p_value, 6),
        "vocabulary_overlap": round(vocabulary_overlap(exemplars, contrasts), 6),
        "n_positive": len(exemplars),
        "n_negative": len(contrasts),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the tightness sweep and write its artefacts.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--permutations", type=int, default=DEFAULT_N_PERMUTATIONS)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    set_global_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    concepts = load_concepts(DEFAULT_DATASET_PATH)
    tight = load_tight_contrasts()

    manifest = build_manifest(
        experiment=output_dir.name,
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        dataset_path=DEFAULT_DATASET_PATH,
        concepts=[concept.concept_id for concept in concepts],
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        repo_root=PROJECT_ROOT,
        extra={
            "conditions": list(CONDITIONS),
            "language": "en",
            "probe_permutations": args.permutations,
            "tight_contrasts_sha256": file_digest(TIGHT_CONTRASTS),
        },
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    engine = CulturalRepE(model_name=args.model, device=args.device, dtype=args.dtype)
    engine.load_model()

    rows: list[dict[str, Any]] = []
    for concept in concepts:
        exemplars = list(concept.examples_en)
        if not exemplars or concept.concept_id not in tight:
            logger.warning("skipping %s: no English exemplars or no tight set", concept.concept_id)
            continue

        sets = {
            "neutral": build_contrast_examples(exemplars, None, language="en"),
            "shipped": list(concept.contrast_en),
            "tight": tight[concept.concept_id],
        }
        for condition in CONDITIONS:
            row = sweep_condition(
                engine,
                concept.concept_id,
                exemplars,
                sets[condition],
                args.permutations,
                args.seed,
            )
            row["condition"] = condition
            rows.append(row)
            logger.info(
                "%s [%s]: overlap %.0f%% -> layer %d, %.3f (p=%s)",
                concept.concept_id,
                condition,
                100 * float(row["vocabulary_overlap"]),
                row["best_layer"],
                row["probe_score"],
                row["p_value"],
            )

    path = output_dir / "tightness.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "concept_id",
                "condition",
                "vocabulary_overlap",
                "best_layer",
                "probe_score",
                "p_value",
                "n_positive",
                "n_negative",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s (%d rows)", path, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
