# Thaqafa-RepE

**Representation Engineering for Arab Cultural Concepts in Large Language Models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/al3obdi/Thaqafa-RepE/actions/workflows/ci.yml/badge.svg)](https://github.com/al3obdi/Thaqafa-RepE/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> *Thaqafa* (ثقافة) means "culture" in Arabic. This project asks whether that
> culture is legible inside a language model's activations, and whether it can
> be steered.

---

## Research Motivation

Alignment research has largely been written in English, by and for a narrow slice
of the world. When a model is asked to reason about *wasta* (الواسطة), *muru'ah*
(المروءة) or the obligations of *diyafa* (الضيافة), it tends to fall back on
translated approximations that flatten the concept into its nearest Western
analogue — nepotism, chivalry, hospitality — and loses the social weight that
makes the original meaningful.

Representation engineering (RepE) offers a way to study this directly. Instead
of probing behaviour through prompts alone, RepE reads the model's internal
activations, isolates the direction that encodes a concept, and then adds that
direction back into the forward pass to steer generation. Thaqafa-RepE applies
this method to Arab cultural concepts in order to answer three questions:

1. **Legibility** — Are Arab cultural concepts linearly represented in the
   residual stream of open-weight instruction-tuned models?
2. **Locality** — At which layers do these representations emerge, and are
   Arabic and English descriptions of the same concept encoded in a shared
   direction?
3. **Controllability** — Can injecting a concept vector make a model's answers
   more culturally grounded without degrading factuality or fluency?

The long-term goal is a reproducible, auditable toolkit for cultural alignment
that other under-represented cultures can adapt, not a single benchmark score.

## Key Features

- **Bilingual concept dataset** — every concept carries Arabic and English
  names, examples, and a note on the cultural context it belongs to.
- **Contrastive concept vector extraction** — read residual stream activations
  at any layer and derive a unit-norm direction per concept via the
  mean-difference recipe (`CulturalRepE.extract_vector`).
- **Curated minimal-pair contrasts** — every concept ships bilingual
  contrast sentences in the same frame with the concept absent; extraction
  and probing prefer them, falling back to a deterministic neutral bank
  (`contrast_ar`/`contrast_en`, `src.data.contrastive`).
- **Steered generation** — inject a concept vector at a chosen strength and set
  of layers via TransformerLens forward hooks, with negative strengths for
  suppression (`CulturalRepE.inject_vector`).
- **Scoped steering** — a `steering()` context manager that removes exactly the
  hooks it added, even when the body raises, so a failed generation cannot leave
  the model silently steered.
- **Evaluation harness** — sweep injection strength and layer sets, measuring
  steering *effect* (KL divergence) against fluency *cost* (cross-entropy)
  (`src.utils.evaluation`).
- **Linear probes** — cross-validated logistic regression per layer, reported
  against the majority-class floor, so the extraction layer is chosen from
  evidence rather than convention (`src.utils.probes`).
- **Prompt-engineering baselines** — direct, persona and Arabic instruction
  framings, so steering is measured against the intervention any practitioner
  would try first (`src.utils.baselines`).
- **Paper scaffold** — a LaTeX skeleton with the methodology written out and
  every unwritten number flagged (`docs/research_paper`).
- **Research-grade tooling** — a committed `poetry.lock`, pinned lint
  tooling, pre-commit hooks, type hints throughout, and CI running ruff,
  black, mypy and pytest with a coverage floor, plus a weekly smoke test
  against real pretrained weights.

> **Status: pipeline complete, experiments pending.** Extraction, injection,
> linear probes, layer-set sweeps and prompt-engineering baselines all work and
> are tested on CPU. What remains is running them on real models and having
> native speakers rate the results — no research findings are claimed yet.

### How extraction works

```text
v = mean(resid_post[layer] | concept prompts) - mean(resid_post[layer] | neutral prompts)
v = v / ||v||₂
```

Subtracting a neutral baseline is what makes the direction concept-specific:
without it the mean is dominated by components shared by every sentence, and
all concepts point roughly the same way. Padding positions are masked out of
the average, each prompt is averaged over its own real tokens before prompts
are averaged together, and the sum is accumulated in float32 even when the
model runs in bfloat16. Extraction defaults to the middle layer, where semantic
features tend to be most linearly separable.

### How steering works

Injection is the mirror image of extraction. A forward hook on the same residual
stream point writes to it instead of reading from it:

```text
resid_post[layer] ← resid_post[layer] + strength · v
```

Because `v` is a unit direction, `strength` is the only magnitude knob: positive
amplifies, negative suppresses, and `0.0` reproduces the unsteered model exactly
— which makes zero a fair baseline measured through the same code path, not a
separate branch. The offset lands on every sequence position, so the whole
context is nudged rather than only the final token.

**Calibrate the strength.** Residual norms grow steeply with depth (61 → 396
across GPT-2's twelve layers), so a fixed absolute coefficient is a strong
intervention early and a negligible one late. Measured on GPT-2 layer 11:

| Mode | Coefficient | Effect (KL) |
| --- | --- | --- |
| absolute | 8.0 | 0.0009 |
| relative | 50% of ‖resid‖ | **0.435** |

```python
engine.calibrate_layer_norms()                      # measure ‖resid‖ per layer
with engine.steering("diyafa_001", strength=0.25, strength_mode="relative"):
    ...                                             # 25% of the layer's norm
```

Use `strength_mode="relative"` for anything compared across layers or models;
`evaluate_layer_sets()` already defaults to it, since comparing configurations
on an absolute grid is not a fair comparison.

Hooks mutate the model until they are removed, and a forgotten hook silently
steers everything that follows it. `steering()` therefore removes exactly the
hooks it added — leaving nested scopes and any caching or ablation hooks you
attached yourself intact — and does so in a `finally`, so an exception mid-
generation cannot leak a live hook.

## Installation

Requires Python 3.10–3.12 and [Poetry](https://python-poetry.org/docs/#installation).

```bash
git clone https://github.com/al3obdi/Thaqafa-RepE.git
cd Thaqafa-RepE

# Install runtime and development dependencies
poetry install

# Install the git hooks (black, isort, ruff, mypy, nbstripout)
poetry run pre-commit install

# Copy the environment template and fill in your own values
cp .env.example .env
```

`.env` is git-ignored. Never commit tokens or credentials — put your Hugging
Face token in `.env` only, and leave `.env.example` empty.

Verify the installation:

```bash
poetry run pytest
poetry run ruff check .
```

## Usage

### Load the concept dataset

```python
from src.data.dataset_builder import load_concepts, filter_by_category

concepts = load_concepts("data/datasets/cultural_concepts.jsonl")
ethical = filter_by_category(concepts, "ethical")

for concept in ethical:
    print(concept.concept_ar, "-", concept.concept_en)
```

### Extract a concept vector

```python
from src.models.rep_engine import CulturalRepE

engine = CulturalRepE(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    device="cuda",
    dtype="bfloat16",
)
engine.load_model()

# Examples come from the dataset by concept_id, the layer defaults to the
# middle of the stack, and the entry's curated minimal-pair contrasts are
# used as the negative side (neutral bank as fallback).
vector = engine.extract_vector("diyafa_001")

# Or supply everything explicitly:
vector = engine.extract_vector(
    concept="diyafa_001",
    examples=["أكرم ضيافته لمدة ثلاثة أيام", "He hosted him generously for three days"],
    contrast_examples=["الطقس جميل اليوم.", "The weather is nice today."],
    layer=14,
)

# Every concept in the dataset at once, then persisted to disk
engine.extract_all_vectors()
engine.save_vectors("outputs/vectors/concept_vectors.pt")
```

Pass the Hugging Face token for gated models through `HF_TOKEN` in your `.env`
(`hf_token=os.environ["HF_TOKEN"]`) — never inline it in code or notebooks.

### Steer generation

Use the context manager — it cleans up after itself:

```python
# Amplify the concept for the duration of the block
with engine.steering("diyafa_001", strength=1.5, layers=[12, 14, 16]):
    steered = engine.model.generate("A guest arrives unannounced.", max_new_tokens=64)

# Suppress it instead
with engine.steering("diyafa_001", strength=-1.5):
    suppressed = engine.model.generate("A guest arrives unannounced.", max_new_tokens=64)

# Hooks are gone here, whether or not the blocks raised
assert engine.active_hook_names == []
```

When the steering has to outlive a block of code, drive the hooks manually:

```python
handles = engine.inject_vector("diyafa_001", strength=1.5)
...
engine.remove_hooks(handles)   # or remove_hooks() to detach everything
```

### Find the right layer with a probe

```python
from src.utils.probes import sweep_layers_with_probe, best_layer

results = sweep_layers_with_probe(engine, "diyafa_001")
for layer, result in results.items():
    print(layer, round(result.accuracy, 3), "vs chance", result.chance)

vector = engine.extract_vector("diyafa_001", layer=best_layer(results))
```

Accuracies are cross-validated and reported against the majority-class floor.
Both matter: with `d_model` features and a handful of prompts a linear
classifier separates almost any labelling of the *training* set, and on an
unbalanced split 0.75 can be worse than always guessing the majority class.

### Measure the strength/fluency trade-off

```python
from src.utils.evaluation import evaluate_steering, summarize_sweep

results = evaluate_steering(
    engine,
    "diyafa_001",
    prompts=["What should I do when a guest arrives unannounced?"],
    strengths=[-2.0, -1.0, 0.0, 1.0, 2.0],
)

for strength, result in results.items():
    print(strength, result.perplexity, result.generations)
```

Each point reports the generated text *and* the model's cross-entropy on the
prompts. The second number is the guardrail: it is measured on text the steering
did not produce, so a sharp rise means the injection is damaging language
modelling rather than merely changing the topic. Plot it with
`plot_steering_sweep(**summarize_sweep(results))` to find the knee — the usable
steering range — instead of hand-picking one coefficient.

### Command line

```bash
# Every concept at the middle layer
poetry run python scripts/extract_vectors.py --output outputs/vectors

# One concept at an explicit layer
poetry run python scripts/extract_vectors.py --concept diyafa_001 --layer 14

# Generate the same prompt unsteered and steered, side by side
poetry run python scripts/inject_concepts.py --concept diyafa_001 --strength 1.5 \
    --vectors outputs/vectors/concept_vectors.pt \
    --prompt "What should I do when a guest arrives unannounced?"
poetry run python scripts/evaluate.py --concept diyafa_001 --min -2 --max 2 --steps 9
```

### Compare against prompt engineering

```python
from src.utils.baselines import compare_steering_vs_prompting

comparison = compare_steering_vs_prompting(
    engine,
    concept="diyafa_001",
    concept_name="Arab hospitality",
    prompts=["What should I do when a guest arrives unannounced?"],
    strength=1.5,
)

for row in comparison.rows():
    print(row)
```

Steering has to beat "answer with Arab hospitality in mind" to justify needing
white-box access to the model. The comparison runs an unprompted control, the
instruction framings and the steered condition on the same prompts, and scores
every condition's *output* with the same unmodified model so the fluency numbers
are comparable. It deliberately names no winner: which answer is more culturally
grounded is a question for human raters.

### Sweep layer sets

```python
from src.utils.evaluation import evaluate_layer_sets, summarize_layer_sets

grid = evaluate_layer_sets(
    engine, "diyafa_001", prompts,
    layer_sets=[[14], [13, 14], [12, 13, 14]],
    strengths=[1.0, 2.0],
)
for row in summarize_layer_sets(grid):
    print(row)   # effect_kl against mean_loss, per configuration
```

Effect and cost are reported separately on purpose. A configuration that leaves
fluency untouched because it did nothing is not a cheap win, and cross-entropy
alone cannot tell the two apart.

## Pilot results

Two complete pilot runs are committed under [`results/`](results/), each with a
`manifest.json` pinning the commit, the seed, the dataset hash and the installed
package versions. Every number the paper quotes comes from an artefact of this
kind — nothing in this repository reports a measurement it did not make.

| Run | Readable above chance (p < 0.05) | Mean balanced accuracy | Adding beats random | Subtracting beats random |
| --- | --- | --- | --- | --- |
| [`gpt2`](results/pilot_gpt2/) (124M, English) | 5 / 12 | 0.681 | 16 / 16 | 16 / 16 |
| [`gpt2-medium`](results/pilot_gpt2-medium/) (355M, English) | 7 / 12 | 0.706 | 28 / 28 | 24 / 24 |
| [`Qwen2.5-0.5B`](results/pilot_qwen2.5-0.5b/) (494M, multilingual) | 11 / 12 | 0.883 | 42 / 44 | 36 / 44 |

The causal columns count only the points measured through a reading probe at
0.70 balanced accuracy or better — a lift through a probe near chance says
nothing, so a model whose probes are weak contributes few points: `gpt2`
supplies sixteen, `gpt2-medium` twenty-eight, Qwen forty-four. The eight subtractions that *fail* in Qwen are left in the table
rather than filtered out: subtraction is not uniformly reversible even where
addition works, and that asymmetry is what the check exists to surface.

**Cross-lingual transfer needs the model to model both languages.** Steering
with a concept's Arabic-only direction and reading with a probe trained
entirely on English gives a median transfer ratio of 0.00 in `gpt2` and 0.00
in `gpt2-medium` — the Arabic direction does nothing an English reader
notices — against **0.62 in Qwen**, where 24 of 39 usable points transfer at
half the ceiling or better. Tripling the parameters within English does not
start it; training on the language does.

That is the same shape as the readability result below, measured a different
way, and it is the sharper of the two: the geometric check found aligned
cosines only ~0.18 above their mismatched controls, which would have
suggested the two languages barely share a direction. Behaviourally they
share most of one.

**`gpt2-medium` is there to separate scale from language coverage**, which the
first two runs confounded. It is the same family, tokenizer and training
distribution as `gpt2` with 2.9× the parameters, so the step from `gpt2` to it
holds language fixed. That step buys +0.025 mean balanced accuracy. The step
from it to a multilingual model of comparable size — 1.4× the parameters —
buys +0.177. Scale within English does something; most of the gap is the
language the model was trained on. That is a three-point comparison on twelve
concepts, not a scaling law.

Every figure above is asserted against the committed CSVs by
`tests/test_readme_numbers.py`, so the table cannot drift from the artefacts
it summarises.

### What the probe scores actually rest on

**This is the most important result in the repository, and it revises the
table above — and corrects an earlier version of this section.**

Extraction subtracts the mean of the contrasts from the mean of the exemplars,
so whatever the two sets share cancels and whatever is unique to one side
survives. The dataset's exemplars and contrasts share about 11% of their
English vocabulary — they are not minimal pairs.

[`results/pair_tightness_qwen/`](results/pair_tightness_qwen/) holds the
exemplars fixed and varies only the negative side, on Qwen2.5-0.5B, in English:

| Contrast set | Overlap | Mean balanced accuracy | Significant (p < 0.05) |
| --- | --- | --- | --- |
| Generic neutral bank | 4% | 1.000 | 12 / 12 |
| The dataset's own contrasts | 11% | 0.849 | 7 / 12 |
| Rewritten to be minimal | 52% | 0.563 | 0 / 12 |

Every concept falls, monotonically. **The reported readability rests
substantially on the exemplars and contrasts differing in subject matter.**

#### But the floor is not evidence of absence

The same run probes properties a language model is known to encode, written as
pairs at least as tight as the concept ones:

| Control | Overlap | Samples | Balanced accuracy | p |
| --- | --- | --- | --- | --- |
| `person` (I / he) | 100% | 12 | 1.000 | 0.015 |
| `negation` (did / did not) | 90% | 12 | 1.000 | 0.015 |
| `tense` (past / future) | 50% | 12 | 1.000 | 0.015 |
| `valence` (delighted / disappointed) | 66% | 12 | **0.500** | 0.512 |
| `valence_large` — *the same contrast* | 58% | **24** | **0.883** | 0.015 |

Three structural controls hit ceiling at overlaps *higher* than the concept
pairs, so tightening does not break a probe. But valence — the control shaped
most like a cultural concept, one evaluative word in an otherwise identical
frame — reads at exactly chance with twelve samples and at 0.883 with
twenty-four. **Same sentences, same method, same model.**

So the honest conclusion is not "these concepts are not linearly represented".
It is:

> At twelve samples per concept, this design detects a difference in subject
> matter and does not detect a single-word semantic difference. Once the topic
> difference is removed, the concepts fall to a floor that a known-encoded
> property also falls to at the same sample size.

The 0.563 is the design's detection floor, not a measurement of the concepts.

#### What this asks of the dataset

Two things, in this order:

1. **More exemplars per concept.** Twelve samples is below what this design
   needs. The valence control puts the working threshold between 12 and 24.
2. **Contrasts that are genuinely minimal** — same frame, same topic, one thing
   changed. Otherwise the score returns to measuring topic.

The tightened contrasts and controls are **English only** and live under
`data/experiments/`, not in the dataset. Writing a minimal pair is experimental
design; judging Arabic cultural content is what native-speaker review is for.

The causal results are less exposed — they inject a direction and read it back
against a matched-norm random control — but the probes reading them were
trained on the same loose pairs and the same twelve samples.

```bash
python scripts/build_review_sheet.py     # docs/review_sheet.md, per concept
python scripts/check_dataset.py          # overlap per entry, and what leaks
python scripts/run_pair_tightness.py     # re-run the experiment above
```

The last column counts only the read-back points measured through a probe that
can actually read its concept (balanced accuracy ≥ 0.70); the rest are reported
in the CSV but should be discarded, and each run's report says so.

```bash
# Reproduce either (CPU, no token required)
python scripts/run_pilot.py --model gpt2 --output-dir results/pilot_gpt2 --seed 42
python scripts/run_pilot.py --model Qwen/Qwen2.5-0.5B \
  --output-dir results/pilot_qwen2.5-0.5b --seed 42
```

The runner probes every layer, extracts each concept at the layer the probe
picked, sweeps norm-relative injection strengths recording effect against cost,
compares steering with prompt-engineering baselines, checks whether a concept's
Arabic-only and English-only directions agree, asks whether steering with a
concept's direction makes that concept's own probe fire on neutral prompts, and
asks the mirror question: whether *subtracting* the direction stops that probe
recognising the concept's own held-out exemplars — and finally whether the
Arabic-only direction steers a probe trained entirely on English, which is the
behavioural version of the cross-lingual question the cosine only answers
geometrically.

Each of those last two carries a control, because the headline number alone
would be uninterpretable: two directions at one layer can be similar for
reasons that have nothing to do with the concept, and a random vector of the
same norm shifts a distribution and moves a probe just as a concept vector
does. What a random direction cannot do is move the *concept's own* probe
further than it moves by chance.

The gap between the two runs is the headline: `gpt2` has almost no Arabic
capability, and it is the concepts' readability that collapses with it, not the
pipeline. `wasta_001` sits at chance in `gpt2` (0.500, p = 0.54) and reaches
0.817 (p = 0.04) in Qwen. Neither run is evidence about Arab cultural concepts
in a deployed model: the exemplar sets are small, the reported layer was chosen
on the same data as its p-value, and most dataset entries still await
native-speaker review. Each run's `README.md` carries those limitations next to
its own numbers.

## Project Structure

```text
Thaqafa-RepE/
├── .github/workflows/ci.yml       # Lint and test on every push and PR
├── notebooks/                     # Exploratory research notebooks
│   ├── 01_data_collection.ipynb   # Build and inspect the concept dataset
│   ├── 02_vector_extraction.ipynb # Extract vectors, sweep layers
│   └── 03_concept_injection.ipynb # Steer generation, sweep strengths
├── scripts/                       # Reproducible CLI entry points
│   ├── run_pilot.py               # Full experiment: probe, steer, baseline
│   ├── build_review_sheet.py      # Lay concepts out for native-speaker review
│   ├── check_dataset.py           # Warnings: pairs that are not minimal
│   ├── extract_vectors.py
│   ├── inject_concepts.py
│   └── evaluate.py
├── results/pilot_gpt2/            # Committed run artefacts + manifest.json
├── src/
│   ├── data/dataset_builder.py    # CulturalConcept dataclass and loaders
│   ├── data/contrastive.py        # Neutral baseline prompts (AR + EN)
│   ├── models/rep_engine.py       # CulturalRepE: extraction and injection
│   ├── utils/baselines.py         # Prompt-engineering comparison conditions
│   ├── utils/evaluation.py        # Strength and layer sweeps, effect vs cost
│   ├── utils/causal.py            # Read-back and suppression, each vs a random control
│   ├── utils/crosslingual.py      # Arabic/English agreement, and steering transfer
│   ├── utils/probes.py            # Cross-validated linear probes per layer
│   ├── utils/provenance.py        # Run manifests: commit, seed, dataset hash
│   └── utils/visualization.py     # Layer sweeps, similarity heatmaps
├── data/
│   ├── raw/                       # Untracked source material
│   ├── processed/                 # Untracked intermediate artefacts
│   └── datasets/                  # Tracked, curated concept dataset (JSONL)
├── tests/                         # pytest suite
├── docs/research_paper/           # LaTeX paper scaffold and build script
├── .env.example                   # Configuration template (no secrets)
├── .pre-commit-config.yaml
├── pyproject.toml                 # Poetry dependencies and tool config
└── CONTRIBUTING.md
```

Only `data/datasets/` is version-controlled. `data/raw/` and `data/processed/`
are ignored so that large or licence-restricted corpora never enter git history.

## Dataset Schema

Each line of `data/datasets/cultural_concepts.jsonl` is one JSON object
(12 concepts currently; thresholds enforced by `tests/test_dataset_integrity.py`):

| Field | Type | Description |
| --- | --- | --- |
| `concept_id` | string | Stable identifier, e.g. `wasta_001` |
| `concept_ar` | string | Concept name in Arabic |
| `concept_en` | string | Transliteration and English gloss |
| `category` | string | `social`, `ethical`, `cultural`, ... |
| `description` | string | One-sentence definition |
| `examples_ar` | string[] | Arabic sentences expressing the concept (≥3) |
| `examples_en` | string[] | English sentences expressing the concept (≥3) |
| `contrast_ar` | string[] | Arabic minimal pairs: same frame, concept absent (≥2) |
| `contrast_en` | string[] | English minimal pairs (≥2) |
| `cultural_context` | string | Where and why the concept matters |
| `sentiment` | string | `positive`, `negative` or `mixed` |
| `dialect` | string | Variety of the Arabic exemplars, e.g. `MSA`, `Gulf` |
| `review_status` | string | `reviewed` or `pending_native_review` |

## Zero-GPU Strategy

Experiments require GPU access (Llama-3-8B has ~16GB of weights), but the
project follows a **hybrid architecture** that avoids owning a GPU:

| Component | Where | Purpose |
|-----------|-------|---------|
| Development & Testing | Local (CPU) | Code, unit tests, lightweight evaluation |
| Vector Extraction | [HF ZeroGPU Space](https://huggingface.co/spaces/al3obdi/thaqafa-repe-extraction) (A10G) | Heavy extraction from Llama-3-8B or Jais-13B |
| Vector Storage | [HF Private Dataset](https://huggingface.co/datasets/al3obdi/thaqafa-repe-vectors) | Shared, versioned vector store |

### How it works

1. **Develop locally** on CPU — all code, tests, and lightweight evaluation run without a GPU.
2. **Extract vectors** on the [ZeroGPU Space](https://huggingface.co/spaces/al3obdi/thaqafa-repe-extraction) — a Gradio UI loads the model on an A10G GPU, extracts contrastive concept vectors, and pushes them to a private HF Dataset.
3. **Load vectors locally** — pull the extracted vectors from the dataset and use them for steering, probing, and evaluation on CPU.

```python
# Load vectors extracted by the Space
from src.utils.hf_integration import load_vectors_from_hf

vectors = load_vectors_from_hf("al3obdi/thaqafa-repe-vectors")
```

See the [Zero-GPU Guide](docs/zero_gpu_guide.md) for step-by-step instructions.

## Roadmap

- [x] **Phase 0 — Scaffolding.** Repository structure, tooling, CI, dataset schema.
- [ ] **Phase 1 — Data collection.** Expand to 100+ concepts with contrastive
      examples, reviewed by native speakers across dialect regions.
- [x] **Phase 2 — Vector extraction.** Contrastive mean-difference extraction
      with masked activation collection and L2 normalisation. Linear probes and
      a reported layer sweep are still open.
- [x] **Phase 3 — Concept injection.** Hook-based injection with correct
      broadcasting, a scoped `steering()` context manager, and a strength sweep
      that reports generations alongside a fluency guardrail.
- [x] **Phase 4a — Evaluation tooling.** Linear probes, layer-set grids,
      prompt-engineering baselines and the LaTeX paper scaffold.
- [ ] **Phase 4b — Running the experiments.** A reproducible `gpt2` pilot with
      committed, provenance-stamped artefacts is done; a capable multilingual
      model, native-speaker rating of cultural grounding, and filling in the
      paper's results are still open.
- [ ] **Phase 5 — Publication.** Release the dataset, vectors and paper.

## Citation

A paper is in preparation. Until then, please cite the repository
(machine-readable metadata in [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{thaqafa_repe_2026,
  title  = {Thaqafa-RepE: Representation Engineering for Arab Cultural Concepts},
  author = {{Thaqafa-RepE Contributors}},
  year   = {2026},
  url    = {https://github.com/al3obdi/Thaqafa-RepE}
}
```

## Contributing

Contributions are welcome — especially new cultural concepts and native-speaker
review of existing entries. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Code is released under the [MIT License](LICENSE). The cultural concepts
dataset (`data/datasets/`) is licensed separately under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) — see
[`data/datasets/README.md`](data/datasets/README.md).
