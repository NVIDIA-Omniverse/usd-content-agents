The Joint Agent Service takes a YAML configuration file (and optionally a USD scene) as input and runs a multi-step VLM-based classification pipeline: build_dataset_usd → prepare_dataset → predict. It produces predictions (JSONL) and an HTML report for each prim in the scene.

## Highlights

* End-to-end pipeline
    * Upload a YAML config file to start the pipeline.
    * Optionally upload a USD file directly (overrides the config's usd_path).
    * Produces predictions.jsonl and an HTML classification report.
* Flexible configuration
    * YAML config controls rendering modes, VLM prompts, and prediction settings.
    * Supports multiple rendering modes (prim_only, composition, etc.).
    * Configurable VLM backend and model.
    * Selectable rendering backend via API parameter:
        * `warp` — In-process CUDA GPU raytracer. Fast (~29ms/frame), no Vulkan/DISPLAY needed.
        * `ovrtx` — Local RTX path tracer via OvRTX subprocess. PBR quality, requires Vulkan.
        * `remote` — Default. Rendering through `RENDER_ENDPOINT`; no local GPU required.
        * Omission uses the configured service default.

* Real-time monitoring
    * Server-Sent Events (SSE) stream provides live updates with:
    * Current step, state, per-step progress, and overall percentage.
    * Automatic reconnection; polling fallback available via status endpoint.
    * Cancel running pipelines via a dedicated endpoint.

* Visual feedback
    * Live preview gallery during the rendering stage.
    * On-demand HTML report generation with per-prim classifications.

* Opt-in Research Preview Joint Rigger
    * An enabled request defaults to the built-in `owned_core` adapter.
    * The normal pipeline supplies Stage 1 predictions plus accepted Stage 2
      candidates and selects an exact contract-derived V1 or V2 request. V2 is
      used for aggregate or multi-root topology; ordinary one-root existing-link
      topology retains V1.
    * Candidate-only CLI/YAML compatibility calls may omit predictions and retain
      transitional Stage 2-only authoring.
    * New outputs are self-contained `cache/joint_rigger/rigged.usdz` packages.
    * The owned path authors topology only; mass and collision flags must be false.
    * The public path selects `owned_core`; it never falls back to an external
      rigger.
