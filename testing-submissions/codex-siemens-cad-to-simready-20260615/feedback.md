# Agent Testing Feedback

## Summary

- Tester: abrasic / Codex
- Coding harness or agent: Codex desktop coding agent
- Date: 2026-06-15
- Asset: /mnt/d/Customer_Files/Siemens/Jun_1_Tests/JT_3D_Product/A5E52945329_P5Z00052019.jt with reference imagery in /mnt/d/Customer_Files/Siemens/Jun_1_Tests/References/A5E52945329
- Workflow attempted: Omniverse CAD-to-SimReady, including JT to USD conversion, Material Agent assignment, Texture Agent generation/application, OVRTX/Windows USD viewer validation, and final USD packaging.
- Final status: Completed with remediation. CAD conversion, material assignment, and texture assignment completed; final USD texture references validated. A packaging issue left WSL-only paths in the first final artifacts, and those paths were fixed to relative paths after user reported an empty reference in Windows.

## What did you make your agent do?

The agent converted a Siemens JT CAD asset to USD, ran the Material Agent using reference imagery, corrected the scene lighting path after black renders were detected, completed a full Material Agent run, then ran the Texture Agent on the materialized USD. After the user noticed that `A5E52945329_P5Z00052019_textured.usd` was only about 134 KB and opened with an empty reference, the agent inspected the USD composition and confirmed that Texture Agent had authored texture opinions but the deliverable needed Windows-portable path normalization. The final flattened USD and lightweight composition layer were then fixed and revalidated.

## What was the running environment with the coding harness?

- Operating system: Windows host with WSL2 Ubuntu 22.04.5 LTS.
- Coding harness and version: Codex desktop coding agent; exact app version not available from the harness.
- Model, if known: GPT-5 based Codex agent.
- Python version: Python 3.12.13 in `/home/abrasic/src/content-agents/.venv`.
- Docker or container runtime version, if used: Docker 29.5.3; Docker Compose v5.1.4.
- GPU and driver details, if relevant: NVIDIA RTX 6000 Ada Generation, driver 581.42 visible in WSL through `nvidia-smi`.
- Render backend: Windows USD Viewer / OVRTX shim path for agent rendering; OVRTX-backed endpoint work was attempted in WSL but needed Windows rendering fallback.
- VLM provider: Content Agents configured provider/NIM path; exact credentials redacted and not copied.
- Install path used: Local CLI from `/home/abrasic/src/content-agents` plus Docker Compose services where available.
- Important environment variables, with secret values redacted: `.env` and generated collection env files were not copied. Logs/configs copied into this submission were redacted for common token/API key/password patterns.

## What worked well?

- JT-to-USD conversion produced a converted USD and validation logs.
- The Material Agent became useful once explicit lighting was added; batch contact sheets gave a fast sanity check for non-black renders.
- The Texture Agent generated and applied texture maps for the discovered materials, with manifests showing completion and no missing generated texture files after retry.
- USD API validation was effective for verifying authored material texture attributes, mesh/material counts, missing texture files, and bad path strings.
- The final remediation made both the lightweight composition layer and flattened USD use relative paths for Windows portability.

## What was hard, frictional, or frustrating?

- The workflow crossed Windows, WSL2, Docker, USD tooling, and viewer/render endpoint boundaries; path handling between `D:/...` and `/mnt/d/...` was a recurring source of failures.
- OVRTX endpoint setup in WSL did not become a clean GPU-backed render endpoint during the session, so the workflow relied on a Windows shim/viewer strategy.
- Initial Material Agent renders were black until scene lighting was added, and black/blank renders were not caught early enough by the default pipeline.
- Texture Agent rendering still produced black final renders in one path even though the USD texture assignments were present, so data-level validation had to substitute for render-level validation.
- One texture generation path returned empty/invalid generation data for `Plastic_Green`, requiring a cached/procedural retry for missing materials.
- The initially delivered `*_textured.usd` looked like the final artifact by name but was a small composition layer with a WSL absolute sublayer; Windows viewers could reasonably open it as empty.
- The flattened USD initially contained WSL absolute texture asset paths and needed manual normalization to relative `textures/...` paths.

## What would your agent propose changing?

- Add a first-class packaging command that emits `*_textured_flat.usd` plus sibling `textures/` with only relative asset paths, and clearly labels lightweight composition layers versus self-contained deliverables.
- Add a post-run validator that opens the final USD, counts meshes/materials, verifies all texture asset paths resolve, fails on `/mnt/*` or drive-specific absolute paths, and prints the recommended file to open.
- Add an automated black-image detector for render batches and fail fast with lighting/camera diagnostics before spending hours on blank outputs.
- Improve WSL/Windows path normalization in Material Agent and Texture Agent outputs, especially for sublayers and `Sdf.AssetPath` texture values.
- Make OVRTX endpoint setup self-checking: confirm NVIDIA Container Toolkit/GPU visibility, render a known colored USD, and report a single actionable failure if GPU rendering is unavailable.
- When NIM/texture generation returns empty base64 or non-image output, surface the material name and retry/fallback path directly in the summary.
- Provide a small non-proprietary CAD demo and reference-image fixture that exercises conversion, material assignment, texture generation, and packaging end to end.

## Artifacts

- Prompt: `prompt.txt`
- Asset: `assets/asset_manifest.txt` explains why the raw CAD/reference assets were not copied.
- Configs: `configs/content-agents-windows-render.yaml`, `configs/texture_agent_discover_config.yaml`, `configs/texture_agent_siemens_a5e52945329.yaml`, `configs/texture_agent_siemens_a5e52945329_retry_missing.yaml`, `configs/environment_snapshot.txt`, `configs/repo_status_before_submission.txt`, `configs/relevant_uncommitted_diff.patch`
- Logs: `logs/usd_convert_cad_status.log`, `logs/usd_convert_cad_validate.log`, `logs/material-agent-stdout.log`, `logs/material-agent-stderr.log`, `logs/texture_agent_run.log`, `logs/texture_agent_retry_missing.log`, `logs/texture_agent_retry_missing_cached_20260614_174905.log`
- Outputs: `outputs/texture_agent_initial_artifacts_manifest.json`, `outputs/texture_agent_initial_uv_report.json`, `outputs/texture_agent_retry_missing_artifacts_manifest.json`, `outputs/texture_agent_retry_missing_uv_report.json`, `outputs/final_usd_validation.txt`
- Screenshots or renders: `outputs/generated_texture_contact_sheet.png`, `outputs/all_textured_albedo_contact_sheet.png`
