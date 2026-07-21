The Material Assignment Service takes a USD scene as input and automatically assigns physically-plausible materials to scene primitives using a vision-language model (VLM), then renders a high-quality “after” image and exports the updated USD. It’s built for repeatable, observable, and explainable results backed by a clear API and a live UI.

## Highlights

* End-to-end pipeline
    * Upload USD, optionally add reference images and a natural-language prompt, then run a multi-step pipeline: build_dataset_usd → prepare_dataset → predict → apply → render.
    * Produces a final USD with materials applied and a final render image.
* Flexible inputs
    * Supports .usd, .usda, .usdc, .usdz.
    * Optional reference images to guide the VLM.
    * Optional generated material-library mode creates asset-specific materials from reference images using deployment image-generation settings.
    * Natural-language prompts can nudge decision-making (e.g., “Metal frames should be aluminum”).
    * Invalid predictions automatically fall back to a canonical neutral gray matte material instead of leaving parts unbound.

* Real-time monitoring
    * Server-Sent Events (SSE) stream provides live updates with:
    * Current step, state, per-step progress, and overall percentage.
    * Step timeline with active/completed indicators.
    * Automatic reconnection; polling fallback available via status endpoint.
    * Cancel running pipelines via a dedicated endpoint.

* Visual feedback and explainability
    * Immediate input render (thumbnail) after upload.
    * Live preview gallery during the rendering stage.
    *  Final, high-res result image and an optional before/after comparison.
    * For predictions, events can include material icons and “reasoning” text showing why a material was chosen.
