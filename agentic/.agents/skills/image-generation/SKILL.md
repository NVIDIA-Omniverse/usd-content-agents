---
name: image-generation
description: Generate or edit raster images for agentic workflows through the coding agent's companion image generator by default or an explicitly configured World Understanding image backend. Use when a workflow needs text-to-image, conditioned image editing, or a recorded generated-image artifact with provider, prompt, input, and output provenance.
metadata:
  author: NVIDIA Omniverse
---

# Image Generation

Generate one recorded raster-image artifact for a calling skill or workflow.
Keep semantic planning and acceptance with the caller; this skill owns provider
selection, image production, and generation provenance only.

## When to Use

Use this skill whenever another agentic skill needs a generated or edited
image, including semantic overlays, material textures, reference images, and
visual proposals. The caller supplies the prompt, conditioning images, output
directory, and the consequences of failure.

## Limitations

- Do not choose semantic workflow fallbacks. Return a completed or failed image
  result and let the caller decide whether to retry, continue differently, or
  stop.
- Do not silently switch providers. An explicit World Understanding backend is
  authoritative. Without one, try only the companion image generator.
- Treat a model named for the companion path as a preference unless the
  companion capability exposes model selection.
- Return raster evidence only. Do not claim that a generated image is a valid
  mask, texture map, material, or final workflow result until the caller checks
  it.

## Prerequisites

Locate the deterministic recorder and explicit-backend adapter:

```bash
if test -f agentic/.agents/skills/image-generation/scripts/generate_image.py; then
  IMAGE_GENERATION_TOOL=agentic/.agents/skills/image-generation/scripts/generate_image.py
elif test -f .agents/skills/image-generation/scripts/generate_image.py; then
  IMAGE_GENERATION_TOOL=.agents/skills/image-generation/scripts/generate_image.py
else
  echo "image-generation tool not found" >&2
  exit 1
fi
```

Write the exact prompt to a UTF-8 text file before generation. Never place API
keys in prompts, command arguments, manifests, or logs.

## Instructions

1. Read the caller's frozen image-generation configuration when one exists.
2. Select exactly one mode:
   - **World Understanding backend:** select this only when the caller
     explicitly supplies a backend. Use `backend` below.
   - **Coding-agent companion:** use this by default when no backend is
     explicitly configured. Invoke the image-generation or image-editing
     capability already exposed to the current agent session.
3. Pass the exact prompt and conditioning images supplied by the caller. For
   editing, preserve any geometry, framing, or registration constraints stated
   in the prompt.
4. Save the raw result without overwriting an earlier attempt.
5. Produce `image_generation.json` with the bundled script. Inspect the image
   and manifest before returning them to the caller.
6. On failure, record a failed result. Do not call a different backend unless
   the caller creates a new request that explicitly selects it.

## Command Reference

Use an explicitly configured World Understanding backend:

```bash
python "$IMAGE_GENERATION_TOOL" backend \
  --prompt-file ATTEMPT/prompt.txt \
  --conditioning-image INPUT.png \
  --output ATTEMPT/raw.png \
  --manifest ATTEMPT/image_generation.json \
  --backend BACKEND \
  --model MODEL \
  --base-url BASE_URL \
  --api-key-env API_KEY_ENV
```

Omit optional model, base URL, API-key environment name, or conditioning
images when the selected registered backend does not need them. The compatibility
backend name `openai_compatible` uses World Understanding's multimodal-chat
image adapter and requires a base URL, model, and endpoint-scoped API key.

For the companion path, first invoke the session's companion image generator.
Ask it to write the result directly to `ATTEMPT/raw.png` when supported. Then
record the result:

```bash
python "$IMAGE_GENERATION_TOOL" record-companion \
  --prompt-file ATTEMPT/prompt.txt \
  --conditioning-image INPUT.png \
  --source-image ATTEMPT/raw.png \
  --output ATTEMPT/raw.png \
  --manifest ATTEMPT/image_generation.json \
  --tool-id companion-image-generation \
  --model MODEL_IF_REPORTED
```

If the companion capability is unavailable or fails before producing an image:

```bash
python "$IMAGE_GENERATION_TOOL" record-failure \
  --prompt-file ATTEMPT/prompt.txt \
  --conditioning-image INPUT.png \
  --manifest ATTEMPT/image_generation.json \
  --mode coding_agent_companion \
  --tool-id companion-image-generation \
  --reason "Concrete failure reason"
```

## Common Workflows

- **Text-to-image:** provide a prompt without conditioning images.
- **Image editing:** provide one or more conditioning images and state what
  must remain invariant.
- **Workflow proposal:** generate the image, then let the caller register,
  project, validate, or reject it using its domain-specific evidence.

## Output Format

Return the generated PNG and `image_generation.json`. The manifest uses
`agentic-image-generation-result.v1` and records:

- `completed` or `failed` status;
- `coding_agent_companion` or `world_understanding_backend` mode;
- provider/tool, backend, requested or reported model, base URL, and API-key
  environment-variable name without the key value;
- prompt and conditioning-image paths and SHA-256 digests;
- output path, SHA-256 digest, dimensions, and image mode; or
- a concrete failure diagnostic.

The caller may embed or reference this manifest in its own evidence. The image
manifest does not decide whether the image is semantically acceptable.

## Troubleshooting

- If the companion tool does not support the requested model, omit the model
  and record the identity it reports.
- If a configured backend rejects conditioning images, report the failure or
  let the caller explicitly revise the request; do not drop inputs silently.
- If an output already exists, create a new attempt directory rather than
  overwriting it.
- If credentials are missing, record the failed explicit-backend attempt and
  report the required environment-variable name without its value.
