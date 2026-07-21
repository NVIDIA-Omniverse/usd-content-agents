---
name: content-workbench
description: Use when a long-running coding agent needs to inspect, manipulate, render, validate, optimize, restore, or export a USD asset through Content Workbench, including scene sessions, snapshots, rendering, picking, path translation, edit transactions, and workflow evidence.
---

# content-workbench

Use this skill when a long-running coding agent needs to inspect, manipulate,
render, validate, optimize, or export a USD asset through Content Workbench.

Content Workbench is the scene and asset state service. The agent owns planning,
workflow choices, evidence review, and user interaction.

## Normal Loop

1. Start or connect to Content Workbench.
2. Discover the API from `/agent-api`, `/agent-api.json`, `/openapi.json`, or
   `/agent/tool-manifest`.
3. Load a local USD scene.
4. Inspect scene hierarchy, candidates, properties, material bindings, and
   diagnostics.
5. Render and pick pixels to ground decisions in visual evidence.
6. Apply reversible preview edits through Workbench APIs.
7. Verify with renders, picks, snapshots, and operation artifacts.
8. Restore/export accepted edits back to source space when required.
9. Preserve artifacts under the run directory.

## Rules

- Do not edit source USD files directly unless the user explicitly asks for
  source-file mutation.
- Prefer Workbench APIs over ad hoc USD scripts for scene state that Workbench
  owns.
- Prefer stable object or revision handles when available; fall back to paths
  only when the current API has not exposed handles yet.
- Treat optimization and restore as Workbench operations, not prompt-time path
  rewriting.
- Record evidence artifacts for every meaningful output claim.
- When the API shape is unclear, fetch the Workbench API docs before guessing.

## References

- `references/agent-workbench-api.md`
- `references/object-identity-and-mapping.md`
- `references/edit-transactions.md`
