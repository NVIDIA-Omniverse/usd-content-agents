# Object Identity And Mapping

The target Workbench contract uses stable handles instead of asking agents to
manually reason over optimizer path maps.

## Target Handles

- `revision_id`: source, optimized, edited, restored, or exported scene state.
- `object_id`: stable Workbench object handle across revisions when mapping
  permits.
- `mapping_id`: durable correspondence between two revisions or path spaces.
- `edit_id`: applied edit transaction.
- `artifact_id`: render, snapshot, report, layer, or trace artifact.

## Current Compatibility Model

Current Workbench sessions expose source and inspection path translation through:

```text
GET  /sessions/{session_id}/optimization
POST /sessions/{session_id}/paths/translate
POST /sessions/{session_id}/paths/translate:batch
```

When an optimized session is used:

- inspection/runtime paths are used for Workbench commands;
- source paths are recovered through path translation;
- ambiguous or unresolved mappings must be recorded as workflow limitations;
- agents must not silently invent source-space targets.

## Agent Rules

- Prefer `object_id` once available.
- If only paths are available, preserve both inspection and source paths in
  artifacts.
- Do not reverse optimizer maps in prompts when Workbench can translate paths.
- Treat ambiguous mappings as explicit workflow evidence.
- Use whole-object edits only when the ambiguity is known and acceptable.
