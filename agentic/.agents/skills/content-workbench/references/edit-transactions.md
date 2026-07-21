# Edit Transactions

The target Workbench edit model is transaction-oriented. Current Workbench
commands are compatibility mechanisms.

## Planned Target Flow

These transaction endpoints are proposed and are not implemented yet. Use the
current compatibility operations below until they are available.

```text
POST /sessions/{session_id}/edits/preview
POST /sessions/{session_id}/edits/apply
POST /sessions/{session_id}/edits/undo
POST /sessions/{session_id}/edits/redo
GET  /sessions/{session_id}/edits
POST /sessions/{session_id}/scene/restore
```

Each edit should return:

- `edit_id`;
- output `revision_id`;
- affected object IDs;
- rejected operations;
- source projection status;
- warnings or unresolved mappings.

## Current Compatibility

Current material and visibility changes go through:

```text
POST /sessions/{session_id}/commands
```

Use compatibility commands when needed, but write artifacts as if edits are
transactions:

- base scene/session;
- target paths or object IDs;
- operation payload;
- result state;
- verification evidence;
- source projection or restore status.

## Restore Rule

Restore/export is a Workbench responsibility. Agents should ask Workbench to
project accepted edits back to source space and should record unresolved mapping
records instead of manually rewriting source paths.
