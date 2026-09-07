<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# How USD Content Agents Are Benchmarked

This guide explains the datasets, methodology, and metrics used to evaluate
USD Content Agents. Use it as a template for creating a benchmark for your own
assets and workflows.

The repository's benchmark implementation and curated evaluation assets are
internal quality-assurance infrastructure. This document describes the
methodology, not the internal execution software.

## Principles

Each benchmark separates two questions:

1. Did the workflow complete the requested task and produce usable evidence?
2. How good was the result according to ground truth or a reference?

The first is a reliability check. The second is a task-quality measurement.
Report both. A good-looking output should not conceal an incomplete run, and a
technically completed run should not be presented as high quality without a
relevant measurement.

Build datasets as versioned manifests with stable case IDs. For every case,
freeze the source asset and any reference image using content digests; record
the request, expected outcome, and asset-family/difficulty tags. Keep withheld
labels and oracle answers out of the agent's working data. When comparing two
runs, hold the case set, asset versions, workflow settings, and model/execution
settings constant.

## Dataset partitions and change control

Use three disjoint partitions even when the workflow changes prompts, skills,
or orchestration rather than model weights:

- The **development set** is the training-like partition. Developers may inspect
  its inputs, outputs, scores, and failures while building general fixes.
- The **validation set** is frozen before a change series begins. Use it to
  compare candidate approaches and select workflow settings, but do not copy
  case-specific answers or repairs into product instructions.
- The **test set** is sealed and owned outside the implementation loop. Run it
  only after the workflow, skills, model, and evaluation configuration are
  frozen for a release decision.

Split by source collection and semantic asset family, not only by random case
rows. Near-duplicate brackets, containers, or parameter variants must remain in
one partition. A public case becomes development data once its prompt,
reference, output, or detailed failure has been inspected during implementation.
Public benchmarks can therefore support reproducible comparison, but cannot by
themselves prove that a foundation model has never seen similar data.

Record the immutable dataset content digest and partition, together with the
digests of the workflow, skills, model settings, provider configuration, and
evaluator in every run. If a sealed-test result
causes any implementation or instruction change, retire that case into the
development/regression corpus and use a fresh sealed set for the next release
claim. Scan product code and instructions for benchmark IDs, hidden dimensions,
copied reference geometry, and case-specific repair recipes before
qualification.

## Workflows benchmarked during development

Below are the Content Agent workflows we benchmark during development.

| Workflow | Dataset | Availability | What it evaluates |
|---|---|---|---|
| Material Assignment | Material regression assets | Internal curated evaluation set | Completion, visible-mesh binding coverage, and optional reference-grounded visual quality across the available execution modes |
| Texture Authoring | Texture evaluation assets | Canonical evaluation set is internal; selected redistribution-cleared examples are public | Scoped preserve/apply/generate texture work, UV handling, visual review, and published output |
| Articulation | Articulation evaluation assets | Internal curated evaluation set | Agentic preparation, independent review, authored joint topology, and final scene evidence |
| Mesh Segmentation | [PartObjaverse-Tiny](https://huggingface.co/datasets/yhyang-myron/PartObjaverse-Tiny) | Public source dataset; internal selection and derived blind evaluation inputs | Face-level semantic segmentation against hidden PartObjaverse-Tiny labels |
| Physics Authoring | PhysX-Mobility | Internal curated evaluation copy and held-out annotations | Per-part physics-property inference across the available execution modes |
| CAD Modeling | CAD modeling evaluation assets | Internal curated evaluation set | Provider-neutral text/image/drawing authoring, revision, parameter families, geometry quality, and Geometry handoff |
| CAD to SimReady | CAD-to-SimReady source assets | Mixed: public Khronos/Thingi10K sources, private NVIDIA prop sources, and an internal curated selection | Source conversion through material/physics authoring, SimReady validation, and final evidence |
| Validation | Validation evaluation assets | Internal curated evaluation set | Validation planning, operation outcomes, issue codes, evidence handling, and terminal assessment |

“Public source dataset” means the original source can be obtained publicly; it
does not mean that the exact benchmark subset, prepared inputs, reference
evidence, or held-out annotations are published. “Internal” means the data is
used only for development and evaluation and is not included in the public
release.

The Validation benchmark has three execution scopes: a provider-free blocking
smoke test with one positive and one expected-negative case; an expanded set
that adds live renderer/judge and judge-unavailable cases; and the complete
nightly/release inventory. The CAD-to-SimReady benchmark currently contains 23
cases spanning NVIDIA prop sources, Khronos GLBs, Thingi10K STLs, and
CAD-converter inputs. The mesh-segmentation benchmark contains 12
PartObjaverse-Tiny cases across all eight dataset categories.

Always retain per-case measurements. A single benchmark average can hide a workflow
that fails a critical asset family or a model that regresses on small parts.

## Materials

### Dataset

Material cases contain a source USD, one or more reference images, the intended
visible scope, and the version of the material library used for authoring.
Select assets that collectively cover painted and unpainted surfaces, metals,
glass or translucent parts, labels/decals, repeated components, multiple
materials on one mesh, and low-quality or ambiguous references. Include enough
views to show the relevant surfaces, but do not expect a reference image to
label every hidden part.

Most material datasets do not have a definitive material label for every prim.
They therefore measure reference-grounded appearance rather than literal
classification accuracy.

### Methodology

First check that the workflow produced a final USD, at least one final render,
and complete material bindings for active visible render-purpose meshes. Report
full bindings, partial material-subset bindings, and completely unbound meshes
separately. This measures delivery completeness, not whether the material
choice is correct.

Then have two independent visual judges compare object-focused final views to
the references. Both judges see the same evidence but not each other's output.
They ignore geometry, camera, crop, background, lighting, and occlusion
differences, and score only mutually visible material evidence. Reject a view
from quality scoring when it is clearly a renderer or shader fallback rather
than a meaningful material rendering.

### Metrics

| Metric | Weight | Meaning |
|---|---:|---|
| Material identity and part mapping | 45% | Correct material class and plausible mapping to visible parts |
| Color palette | 25% | Color correctness and separation between parts |
| Surface finish | 20% | Roughness, gloss, metalness, transparency, and texture character |
| Material consistency | 10% | Consistent treatment across views and repeated/equivalent features |

Each judge assigns 0–100 to every metric. Its overall score is the weighted
sum; the reported score is the arithmetic mean of the two judges. Also report
confidence, rationale, visible issues, and inter-judge agreement. For one
metric, agreement is `100 - |judge A - judge B|`.

Useful reporting thresholds are an **acceptable-rate** share of cases at
overall score ≥70 and a **high-fidelity-rate** share at ≥85. These are
operational baselines, not ground truth. Before turning either threshold into a
release gate, calibrate it against a stratified human-reviewed set and measure
judge-to-human agreement and repeatability.

Both rates use every selected case as their denominator. A case with an
incomplete workflow or no valid quality score does not count toward either
threshold and is also reported separately as incomplete or unscored; it is
never omitted from the rate.

## Texture authoring

### Dataset

Texture cases specify the source asset, request, target material or prim scope,
reference images when applicable, and UV policy. Include four important groups:

- preserve-only cases, where existing approved texture work must not change;
- apply cases, where a provided texture must be bound to an explicit scope;
- generate cases, where the requested appearance must be created and applied;
- UV-readiness cases, including valid UVs, missing UVs, and assets that require
  source preparation before texture work is possible.

Use multi-unit assets so that scope mistakes are measurable. Include cases in
which some units should change while others must remain untouched.

### Methodology and metrics

Score whether every requested unit reaches the expected outcome: published,
preserved, rejected with an explained reason, or deliberately waiting for a
separate generation step. A paused generation handoff is not a published
success, but it can be the correct result for a benchmark case designed to test
that handoff.

Measure scope exactly: which materials/prims were selected, which action was
taken per unit, and whether non-target units changed. Inspect the saved USD,
not merely a plan or preview. Require current final views to be tied to the
evaluated USD; otherwise a stale preview can create a false pass.

There is intentionally no universal texture-quality number. Report visual
review results, target-unit completion rate, scope precision/recall when a
ground-truth scope exists, UV-preparation success rate, and the count of
unresolved units. If references support an appearance comparison, score only
the visible target surfaces and label it as reference-grounded quality.

## Mesh segmentation

### Dataset

The canonical segmentation benchmark uses the public
[PartObjaverse-Tiny](https://huggingface.co/datasets/yhyang-myron/PartObjaverse-Tiny)
dataset (CC-BY-NC-4.0). It uses 12 assets across all eight PartObjaverse-Tiny
categories, with 3–11 semantic parts per asset. The selected corpus includes
balanced controls, thin-part cases, repeated-part cases, and a higher-face-count
stress case; it is intentionally more diagnostic than a random sample.

PartObjaverse-Tiny supplies a GLB and one semantic label for each source
triangle. Benchmark preparation verifies exact triangle/label alignment,
concatenates source primitives in source order, and welds coincident vertices
without deleting or reordering triangles. The resulting blind USD therefore
preserves exact face-level ground truth while exposing only geometry to the
agent. The labels, their face locations, and derived answer files remain
evaluation-only. An ontology-guided track can supply the valid part names; an
open-ended track omits even that vocabulary.

### Methodology and metrics

Require a segmented USD and valid final render evidence. Before calculating a
score, verify prediction and ground-truth face counts: a label file that no
longer corresponds to the source mesh invalidates the comparison.

Calculate:

| Metric | Formula | Why it matters |
|---|---|---|
| Face accuracy | correctly labeled faces / all labeled faces | Overall labeling correctness |
| Per-class IoU | intersection of predicted and true faces / their union | Quality for one semantic class |
| Macro IoU | mean of per-class IoU | Gives small and rare classes equal weight |

The current baseline treats face accuracy below **0.80** and macro IoU below
**0.65** as warnings while establishing model baselines. Preserve the raw
metrics, per-class IoU, and confusion matrix. Accuracy alone can look strong
when a benchmark contains a large dominant part and the model consistently
misses small parts.

## Physics authoring

### Dataset

The canonical property benchmark uses curated
**PhysX-Mobility** objects. Each source object supplies a converted USD and
per-part physics annotations, including material type and density. The
benchmark maps detailed material names into coarse material families (such as
metal, plastic, rubber, and wood) for classification, and derives the relevant
friction/restitution reference values from the annotated material data. Unknown
or intentionally unscored parts remain explicit in the evaluation data rather than
being silently removed from scoring.

The input USD is deliberately physics-stripped before the workflow sees it:
mass, density, friction, restitution, inertia, and physics-material bindings
are removed, while rigid-body and collider structure are retained so component
inference remains meaningful. This prevents the workflow from reading the
values it is being asked to infer. The benchmark also renders a reference image
from that curated asset and records its provenance.

Retaining structure while stripping values is essential, not cosmetic. Physics
component inference groups parts by rigid bodies, colliders, and joints, so an
input stripped of that structure collapses into a single component per asset.
The workflow then makes one material/property decision per asset against a
ground truth with several labeled parts, which caps per-part accuracy at the
component count regardless of inference quality — and the cap is invisible in
an aggregate score. Verify granularity explicitly: record the number of
components the workflow could distinguish alongside the number of labeled
parts, and treat a collapsed decomposition as a dataset-preparation defect,
not an inference failure.

Both the fixed-pipeline and agentic benchmarks run the same PhysX-Mobility case
set, held-out annotations, and property scorer, so their results are directly
comparable. Pin the executing model and agent-runtime version explicitly in
every run and verify the pin in the recorded request metadata; an unpinned
default can roll over between runs and silently change what a comparison
measures. Keep property inference separate from behavior-target tuning:
fitting friction or restitution to a desired motion is not a valid measurement
of whether the workflow inferred those properties from the asset.

### Methodology and metrics

Require a final physics USD and predictions for each eligible part. Report parts
with effectively zero mass: they are a practical simulation risk even when
other fields are present. Track label coverage so a result cannot improve its
score by omitting hard parts.

| Metric | Calculation | Interpretation |
|---|---|---|
| Material-family accuracy | correct material families / labeled parts | Main thresholded classification metric |
| Density error factor | scale-aware ratio/error between prediction and label | More meaningful across large density ranges |
| Friction error | absolute predicted-minus-label error | Continuous regression measurement |
| Restitution error | absolute predicted-minus-label error | Continuous regression measurement |
| Ground-truth coverage | scored labeled parts / eligible labeled parts | Detects missing predictions |
| Zero-mass count | number of output bodies at or below the configured minimum | Simulation-safety diagnostic |

The current material-family baseline is **0.50** accuracy. Density,
friction, and restitution are reported as distributions (for example median,
mean, and spread), not reduced to a universal pass/fail band. Mass is recorded
but not used as a ground-truth metric when it should be calculated by the
workflow from density and volume. For agentic property-inference runs, record
that behavior tuning was disabled. If the workflow includes simulation
validation, a resolved simulation-ready result is a primary outcome and visual
or behavior concerns should be reported separately.

When interpreting continuous-property errors, separate authoring policy from
inference error before drawing conclusions. A metric that stays flat across
model, prompt, and input interventions usually indicates a fixed workflow
default or an upstream bottleneck rather than a perception failure — for
example, a workflow that authors near-zero restitution by policy will show a
large, intervention-invariant restitution error against material-derived
references even when its material classification is correct. Classify such a
gap as a policy difference only when execution evidence shows that the workflow
authored a fixed default; then the corrective action is a default change, not a
model change. Otherwise report it as an unresolved upstream bottleneck and
investigate the responsible component before assigning a cause.

### Runtime behavior

Property inference is complemented by a runtime drop-test benchmark: each
authored asset is placed in a standardized drop-and-settle scenario and must
reach a stable rest state within a bounded observation window. Acceptance first
requires evidence that the simulation advanced through the scenario; reject a
paused or disabled simulation even if its initial pose happens to meet the
geometry gate. The final state must have bounded linear and angular velocity
and the asset must settle (pose unchanged for a fixed duration). The geometry
gate is scale-relative: residual interpenetration must be below the larger of a
small absolute floor and a fixed fraction of the asset's bounding-box diagonal, so
millimeter-scale and meter-scale assets are judged by the same standard.
Compute that diagonal from the asset's fully resolved world-space bounds; a
bounds measurement that ignores an authored scale transform silently converts
the relative threshold into a near-zero absolute one.

Report outcome categories separately — accepted, gate-rejected, stopped, budget
exhausted, and infrastructure/tool failure — rather than one pass rate, because
they have different fixes. Two operational requirements keep those categories
honest. First, any per-step or per-agent-turn timeout must exceed the scenario
deadline it wraps, or the harness kills runs mid-evaluation and inflates the
tool-failure count with results that succeed verbatim on retry. Second, when an
external optimization service participates in tuning, verify before the run
that every tunable parameter binds an attribute the workflow actually authors
(a parameter requiring an authored value the workflow never writes resolves
against nothing and aborts the whole tune), treat transient service errors as
retryable rather than deterministic failures, and judge success from the
produced artifacts rather than process exit codes.

## Articulation

### Dataset

Articulation cases include source and dependency identities, the permitted
candidate scope, a review policy, and—where an oracle is available—an expected
joint graph. The graph should define joint endpoints, type, axes, limits,
ownership, and expected joint count. Include accepted examples, legitimate
rejections, and known non-success cases. Store any oracle graph and post-review
answer outside the agent's writable workspace.

### Methodology and metrics

Evaluate both the topology and the decision process. Confirm that proposed
candidates stay within scope; that review is independent of the proposal; that
the authored USD agrees with the reviewed decision; and that the final scene
evidence comes from the authored output.

For a ground-truth graph, report exact-match rate and component metrics for
endpoints, joint type, axes, limits, owner, and count. A file that contains
some joints is not sufficient. For known-negative cases, success means reaching
the correct non-success outcome with the correct issue history, rather than
fabricating an approved articulation.

## CAD modeling

### Dataset

CAD modeling cases begin with text, reference images, technical drawings, or a
combination of those inputs. Authoring runs use an explicitly selected external
provider, and provider execution remains inside that provider's authorized
environment. Geometry Agent receives an immutable, digest-bound source bundle,
processes only exported representations, and never imports or executes retained
provider-native source.

Use two complementary case groups. **Authoring cases** measure generation,
revision, and parameter-family behavior. **Handoff controls** begin with frozen
known-good and known-bad CAD, mesh, or USD packages and isolate source admission,
units and axes, provenance, validation, USD preparation, and evidence generation
from authoring quality. A handoff control that passes does not establish
text-to-CAD quality.

Stratify authoring cases across dimensioned mechanical parts, curved and
lofted forms, shells and openings, repeated features, multipart objects,
semantic revisions, and parameter families. Include legitimate rejection and
unsupported-capability cases. Each case records the exact request and inputs,
the applicable provider capabilities, required artifact formats, stated
dimensions and tolerances, expected parts and functional features, and any
authoritative reference geometry withheld from the authoring process.

Score only requirements supported by the supplied input. When a prompt omits a
dimension or allows several valid constructions, do not treat similarity to one
hidden reference as ground truth for that unspecified choice. Preserve the
ambiguity in the case definition and use reference-shape measurements only for
features that the input determines.

### Methodology and metrics

Evaluate authoring and Geometry handoff as separate stages. Where the provider
surface permits it, compare a provider-native/default run and one delegated
authoring request with the input, provider version, model, reasoning settings,
requested formats, attempt budget, and time limit held constant. Evaluate the
complete skill-guided workflow separately under a pre-registered revision and
repair budget; its additional orchestration is an intentional system difference
that must remain visible in the efficiency results. When a native system has an
embedded model or unavailable settings, label the result as a system-level
comparison rather than attributing the difference to skills.

Pre-register provider eligibility from its advertised capabilities. Report an
unsupported case as capability coverage, not as a geometry score and not as a
case silently removed after execution. Run any CAD-kernel or benchmark-native
reference evaluator in an isolated evaluation environment; its presence does
not imply that the kernel or authoring runtime ships with Geometry Agent.

Keep first-attempt results separate from results after bounded revision or
repair. Score the exact exported artifact selected by the run, then convert
that same artifact through the common Geometry handoff. Final visual evidence
must come from the resulting USD through OVRTX, with the source/output digest
and render metadata retained. Human reviewers receive the exact request and
applicable references but should remain blind to the provider and comparison
lane whenever the evidence permits it.

| Metric | Meaning |
|---|---|
| Authoring completion | Requested native or exported geometry artifact was produced and was readable without an undeclared fallback |
| Requirement fidelity | Stated dimensions, counts, openings, clearances, placement, and part relationships satisfy their case-specific tolerances |
| Geometric validity | Outputs are readable and have the expected body/component count, topology, connectivity, and manifold or watertight properties where required |
| Reference agreement | Dimension error and applicable shape-distance, overlap, or topology measurements against an authoritative reference |
| Semantic editability | Declared parts, parameters, bounds, units, revision lineage, and requested parameter variants are present and effective |
| Handoff integrity | Typed immutable source metadata, artifact identity, units, axes, provenance, USD validation, and final evidence remain bound to the selected source |
| Visual review | Prompt-aware human review of OVRTX views for proportions, feature fidelity, continuity, floaters, clipping, and view consistency |
| Efficiency | Wall time, model time, token use, attempt count, and bounded revision/repair count |

Report the metric vector and per-case outcomes rather than collapsing every CAD
claim into one score. A valid export can still miss the requested design, and a
visually similar mesh can still lack the claimed parameters or topology. CAD
modeling success also does not establish materials, collision quality, physics,
articulation, manufacturing readiness, or SimReady conformance; those claims
belong to their corresponding workflows and validators.

## CAD to SimReady

### Dataset

The canonical CAD-to-SimReady corpus has **23 source-asset cases**. It combines
NVIDIA prop sources, Khronos glTF/GLB assets, and Thingi10K STL assets, plus
STEP, STP, and SLDASM inputs that exercise the CAD converter. This is a source
conversion benchmark, not a benchmark that starts from an already prepared
USD: each case begins with the original supported source format and follows the
same end-to-end workflow.

Every case records the source file and SHA-256, collection, format, expected
stage sequence, whether CAD conversion is required, and source unit scale. The
required sequence is always `convert → material → physics → validation`. The
benchmark uses the Prop-Robotics-Neutral SimReady profile and convex-hull
collision approximation. Final evidence uses a hero render, multi-view render,
24 ordered turntable PNG frames, and a Pillow-generated animated GIF at 12.5
fps from the validated USD.

The corpus deliberately includes format and units edge cases. STL files do not
store linear units, so the benchmark uses explicit overrides: most NVIDIA-prop STL
sources are treated as meters, the NVIDIA centrifuge is millimeter-authored,
and Thingi10K STL sources are millimeter-authored. The Khronos GearboxAssy and
Lantern cases also receive a longer simulation observation window because their
declared units and body sizes require more time to reach the unchanged settling
criteria. These overrides are part of the benchmark definition, not
per-run tuning.

### Methodology and metrics

This is an end-to-end benchmark: conversion, material authoring, physics
authoring, validation, and final rendering must be evaluated together. Measure
stage completion rate, profile-validation pass rate, conversion success rate,
and final-evidence completeness. Preserve failed profile feature and
requirement identifiers, not only a Boolean result; they reveal recurring gaps
in conversion or authoring.

Require the final views and animation to come from the exact USD that passed
validation. Report resource use—wall time, agent-stage time, and model-token
usage—as a separate efficiency series. Do not replace formal profile validation
with a visual review, or treat a visually plausible output as proof of
simulation readiness.

## Validation Agent

### Dataset

Validation is benchmarked with both good and bad inputs. Freeze the expected
plan/order of operations, allowed statuses and issue codes, source identity,
and expected final disposition. The dataset should contain:

- a known-valid source that should pass;
- known-invalid sources that should surface specific issues;
- renderer or judge dependency-unavailable cases;
- profile-validation positives and negatives; and
- cases with bounded, explicitly documented issue-code variability.

### Methodology and metrics

The central metric is **exact expected-outcome accuracy**: whether the workflow
performed the intended checks and reached the expected disposition. Report
plan-match rate, operation-outcome accuracy, issue-code precision/recall where
the expected codes are known, false-pass rate, false-fail rate, and the rate at
which unavailable dependencies are reported as unavailable rather than passed.

Validate current visual evidence separately. An image is current evidence only
when it comes from the evaluated source/output and renderer invocation. Generic
references, diagnostics, and stale previews must never cause a validation pass.
An expected-negative case is scored as correct when it fails for the expected,
evidenced reason.

## Building your own benchmark

1. Define the claimed workflow capability in one sentence, then choose the
   smallest metrics that demonstrate it.
2. Build a versioned, representative dataset with positive, negative, boundary,
   and historical-regression cases. Stratify by asset family and difficulty.
3. Keep ground truth, reference data, and the source asset immutable. Withhold
   answer labels from the agent whenever possible.
4. Record an individual result for every case: completion outcome, raw metrics,
   configuration, and the final output/reference evidence needed to inspect it.
5. Publish distributions and per-case tables, not only a mean. Include coverage
   and exclusions so missing hard cases cannot improve the score.
6. Add a fast smoke tier first, then integration and nightly tiers as the
   dataset grows. Do not compare results across changed dataset or evaluator
   versions without labeling the comparison as a new configuration.
7. Calibrate any model-assisted judge against human review before using it as a
   gate. Measure agreement, repeated-run variance, and behavior on ambiguous
   cases.

A benchmark result should state the dataset version, workflow version,
configuration, number of cases, verdict breakdown, primary metrics, and known
limitations. A passing benchmark establishes performance on the measured case
set; it does not guarantee success on every asset or prompt.
