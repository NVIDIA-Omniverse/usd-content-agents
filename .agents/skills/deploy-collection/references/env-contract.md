# Environment Contract

The collection deployment writes non-secret runtime values to
`deploy/collection/.collection.generated.env`. API keys remain in the repo-root
`.env` file.

**DO NOT COMMIT `.env`.** Keep the repo-root `.env` gitignored because it
contains API keys and tokens; share placeholders or setup instructions instead.

## Shared Render

- `RENDER_ENDPOINT`: OVRTX or compatible render service URL.
- `MA_RENDERING_USE_DATA_URI=true`: Material Agent sends render payloads
  directly to local/external OVRTX.
- `PA_RENDER_BACKEND=remote`: Physics Agent uses `RENDER_ENDPOINT`.
- `WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS`: caps render request concurrency.

## Material Agent

- `MA_VLM_BACKEND`, `MA_VLM_MODEL`, `MA_VLM_NIM_BASE_URL`
- `MA_VLM_TEMPERATURE`, `MA_VLM_MAX_TOKENS`
- `MA_LLM_BACKEND`, `MA_LLM_MODEL`, `MA_LLM_NIM_BASE_URL`
- `MA_LLM_TEMPERATURE`, `MA_LLM_MAX_TOKENS`
- `MA_IMAGE_GEN_BACKEND`, `MA_IMAGE_GEN_MODEL`, `MA_IMAGE_GEN_BASE_URL`
- `MA_IMAGE_GEN_API_KEY`
  - Used by Material Agent-generated reference images and Material Agent
    generated material-library textures. It may point at the same endpoint/key
    as Texture Agent texture generation.
- `MA_CLUSTER_EMBEDDING_BACKEND`, `MA_CLUSTER_EMBEDDING_MODEL`
- `MA_CLUSTER_EMBEDDING_BASE_URL`, `MA_CLUSTER_EMBEDDING_API_KEY`

## Physics Agent

- `PA_VLM_BACKEND`, `PA_VLM_MODEL`, `PA_VLM_NIM_BASE_URL`
- `PA_VLM_TEMPERATURE`, `PA_VLM_MAX_TOKENS`

## Texture Agent

- `TA_IMAGE_GEN_BACKEND`, `TA_IMAGE_GEN_MODEL`, `TA_IMAGE_GEN_BASE_URL`
- `TA_IMAGE_GEN_API_KEY`
- `TA_LLM_BACKEND`, `TA_LLM_MODEL`, `TA_LLM_BASE_URL`

## Endpoint Notes

- Self-hosted OpenAI-compatible VLM/LLM endpoints should usually use backend
  `nim` with a `/v1` base URL.
- Texture Agent texture generation, Material Agent-generated reference images,
  and Material Agent-generated material-library textures use backend `openai`
  when the sidecar exposes an OpenAI-compatible image API.
- Use `not-used` only for explicit no-auth local endpoints.
