# Command Reference

All commands share these global options:

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--verbose` | `-v` | False | Enable DEBUG logging |
| `--log-file` | | None | Path to log file |
| `--log-level` | | INFO | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `--version` | `-V` | | Show version and exit |

---

## material-agent run

Execute the full multi-step pipeline. **This is the primary command.**

```bash
material-agent run <config> [OPTIONS]
```

| Parameter | Kind | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `config` | Argument | Yes | | Path to unified YAML config |
| `--skip` | Option | No | None | Comma-separated steps to skip |
| `--only` | Option | No | None | Comma-separated steps to run exclusively |
| `--session-id` | Option | No | None | Reuse existing session ID |
| `--resume` | Option | No | False | Resume from last checkpoint |
| `--dry-run` | Option | No | False | Show plan without executing |
| `--clean` | Option | No | False | Delete working dir before starting |

---

## material-agent configure

Create a new pipeline configuration interactively.

```bash
material-agent configure <output_config> [OPTIONS]
```

| Parameter | Kind | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `output_config` | Argument | Yes | | Path to output YAML config to create |
| `--materials-manifest` | Option (`-m`) | No | None | Path to materials manifest YAML |
| `--reference-image` | Option (`-r`) | No | None | Reference image path (repeatable) |
| `--force` | Option (`-f`) | No | False | Overwrite existing config |

---

## material-agent benchmark

Run predictions and evaluate with LLM-judge scoring. Produces FCS metrics.

```bash
material-agent benchmark <config> [OPTIONS]
```

| Parameter | Kind | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `config` | Argument | Yes | | Path to YAML config |
| `--dataset` | Option (`-d`) | No | None | Override dataset path |
| `--output` | Option (`-o`) | No | None | Override output directory |
| `--resume` | Option | No | False | Resume from existing predictions |
| `--stream-predictions` | Option | No | True | Stream predictions to file as produced |

---

## material-agent predict

Run VLM prediction only. Alias for `material-agent run <config> --only predict`.

```bash
material-agent predict <config>
```

| Parameter | Kind | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `config` | Argument | Yes | | Path to unified YAML config |

---

## material-agent apply

Apply predicted materials to USD only. Alias for `material-agent run <config> --only apply`.

```bash
material-agent apply <config>
```

| Parameter | Kind | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `config` | Argument | Yes | | Path to unified YAML config |

---

## material-agent evaluate

Evaluate existing predictions against ground truth using an LLM judge.

```bash
material-agent evaluate <config> [predictions]
```

| Parameter | Kind | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `config` | Argument | Yes | | Path to evaluation config |
| `predictions` | Argument | No | None | Path to predictions JSONL (overrides config) |

---

## material-agent refine

Iterative predict-apply-render-judge loop. Repeats until judge approves or
max iterations reached.

```bash
material-agent refine <config> [OPTIONS]
```

| Parameter | Kind | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `config` | Argument | Yes | | Path to YAML config |
| `--max-iterations` | Option (`-n`) | No | None | Override max iterations from config |

---

## material-agent build-dataset usd

Build a dataset by rendering views of each prim in USD file(s).

```bash
material-agent build-dataset usd <config> [OPTIONS]
```

| Parameter | Kind | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `config` | Argument | Yes | | Path to data preparation config |
| `--source` | Option (`-s`) | No | None | USD file or directory (overrides config) |
| `--output` | Option (`-o`) | No | None | Output directory (overrides config) |
| `--extract-metadata` | Option | No | False | Extract prim metadata |

---

## material-agent build-dataset pdf_vectorstore

Build a multimodal vector store for retrieving specification evidence from PDF
documents.

```bash
material-agent build-dataset pdf_vectorstore <config> [OPTIONS]
```

| Parameter | Kind | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `config` | Argument | Yes | | Path to YAML config |
| `--source` | Option (`-s`) | No | None | PDF file or directory (overrides config) |
| `--output` | Option (`-o`) | No | None | Output directory (overrides config) |

---

## material-agent build-dataset prepare-dataset

Prepare dataset with CMF specifications for prediction or benchmarking.

```bash
material-agent build-dataset prepare-dataset <config> [OPTIONS]
```

| Parameter | Kind | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `config` | Argument | Yes | | Path to YAML config |
| `--vector-store` | Option | No | None | Override vector store path |
| `--dataset` | Option (`-d`) | No | None | Override dataset path |
