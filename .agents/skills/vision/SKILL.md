---
name: vision
description: Analyze images using a Vision-Language Model through the wu CLI. Use when the user wants to describe an image, caption a picture, ask questions about visual content, inspect renders or screenshots, compare visual evidence, or produce text or JSON answers with public VLM backends.
version: "0.1.0"
author: NVIDIA Content Agents
tags:
  - content-agents
  - vision
  - vlm
  - cli
tools:
  - Shell
  - Filesystem
  - wu
compatibility: Requires the wu CLI, readable image files, and provider credentials for the selected VLM backend such as NVIDIA_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY.
---

# Vision

Analyze images with `wu vision` and a configured VLM backend.

## When to Use

- Use when the user asks to describe, caption, inspect, or ask questions about
  an image.
- Use when the user wants JSON output for downstream automation.
- Use when a render, screenshot, or reference image needs quick VLM feedback.
- Use the repository image attachment viewer first when the user needs visual
  inspection inside this conversation rather than a CLI/backend call.

## Limitations

- The command sends image content to the selected model provider. Confirm that
  the user is comfortable with that provider before sending sensitive images.
- Backend behavior and image limits vary by provider and model.
- The CLI analyzes one image path per invocation.
- JSON output is the tool response shape, not a guaranteed strict schema for
  arbitrary prompts.

## Prerequisites

- Activate the repo Python environment and confirm `wu` is on `PATH`.
- Ensure the image file exists locally.
- Set the credential for the selected backend:
  `NVIDIA_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`.

## Instructions

1. Confirm the image path and the user's question or captioning goal.
2. Choose the backend. Use `nim` by default unless the user asks for another
   supported provider.
3. Use a specific `--model` only when the user requests it or the task needs a
   known model.
4. Use `--format json` for programmatic handoff.
5. Summarize the VLM answer and include the exact command used.

## Command Reference

```bash
wu vision <image_path> [OPTIONS]
```

| Option | Description |
|---|---|
| `--prompt`, `-p` | Prompt or question. Default asks for a detailed description. |
| `--backend`, `-b` | VLM backend: `nim`, `gemini`, `openai`, or `anthropic`. |
| `--model`, `-m` | Optional model name. |
| `--system-prompt` | System prompt for the VLM. |
| `--temperature`, `-t` | Sampling temperature. Default is `0.7`. |
| `--max-tokens` | Maximum response tokens. Default is `1024`. |
| `--format`, `-f` | `text` or `json`. |
| `--verbose`, `-v` | Enable debug logging. |

## Common Workflows

```bash
# Describe an image.
wu vision image.png

# Ask a specific question.
wu vision image.png -p "What objects are visible in this image?"

# JSON output.
wu vision image.png --format json

# Use OpenAI.
wu vision image.png --backend openai

# Use a specific public NIM model.
wu vision image.png --backend nim --model "google/gemma-4-31b-it"
```

## Output Format

Report:

- Command executed, image path, backend, model when provided, and output
  format.
- The answer or a concise summary of the answer.
- Any confidence caveats from the model output.
- Any blocker such as missing file, missing API key, unsupported backend, or
  provider timeout.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| API key required | Credential for the selected backend is missing. | Set the matching provider key in the local environment or `.env`. |
| File not found | Image path is wrong or relative to another directory. | Use an absolute path or rerun from the correct working directory. |
| Slow response | Large image, cold provider, or high token budget. | Try a smaller image, lower `--max-tokens`, or another backend. |
| Weak answer | Prompt is too broad or the image lacks detail. | Ask a more specific question or provide a higher-quality image. |
