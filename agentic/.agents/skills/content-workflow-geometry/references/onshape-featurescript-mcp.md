# Onshape Labs FeatureScript MCP

Use the official MCP endpoint only:

```text
URL: https://fs-mcp.labs.onshape.app/mcp
Transport: HTTP
Authentication: user-completed Onshape OAuth
```

The user must subscribe to the Onshape Labs application and authorize it in
their MCP-compatible client. Never ask for, store, forward, or reuse an Onshape
OAuth token. Never ask the user to disclose an API key or secret value. The
optional local export helper may read API credentials that the user configured
outside the conversation; Geometry Agent service does not receive them and does
not host this MCP server.

Before authoring, discover the current tools and read the FeatureScript notes.
For new code, call `test_featurescript` with a trivial lambda and use the returned
library version. Validate the complete feature with `test_feature` before calling
`create_geometry`. Give every exposed parameter a default value. Preserve the
version header when editing existing FeatureScript.

The current MCP surface creates and tests FeatureScript geometry but does not
export or download CAD files. Prefer a manual export when the user does not need
an unattended continuation.

## Optional API Export

When the user asks the agent to continue automatically, offer exactly two
choices: export the file manually, or configure a personal Onshape API key for
the local export helper. Never ask the user to paste a key or secret into chat.
Use wording equivalent to:

> To continue automatically, configure `ONSHAPE_API_KEY` and
> `ONSHAPE_API_SECRET` in the local execution environment or secret manager. Do
> not paste either value here. Tell me only when they are configured. Otherwise,
> export STEP or glTF manually and provide the file.

The compatible secret name `ONSHAPE_SECRET` is also accepted. Do not use command
arguments for credentials, print the environment, enable shell tracing, write a
credential file, or copy credentials into prompts, logs, manifests, source
packages, evidence, or subprocess output. The helper uses HMAC-signed requests,
disables ambient proxies and redirects, bounds every response, and emits
secret-free errors.

MCP-created geometry is normally in a mutable workspace. Ask for confirmation
before creating a named immutable version, then run:

```bash
geometry-agent export-onshape \
  --document-id DOCUMENT_ID \
  --workspace-id WORKSPACE_ID \
  --snapshot-name "Geometry Agent export YYYY-MM-DD" \
  --element-id ELEMENT_ID \
  --element-kind partstudio \
  --format step \
  --rights-assertion "User-authorized export from the user's Onshape document." \
  --output-dir runs/onshape-step
```

For an existing immutable version, replace `--workspace-id` and
`--snapshot-name` with `--version-id VERSION_ID`. Export STEP for exact CAD
handoff. A second glTF export from the same version is useful for public USD and
OVRTX workflows when an exact B-rep verification backend is unavailable:

```bash
geometry-agent export-onshape \
  --document-id DOCUMENT_ID \
  --version-id VERSION_ID \
  --element-id ELEMENT_ID \
  --element-kind partstudio \
  --format gltf \
  --rights-assertion "User-authorized export from the user's Onshape document." \
  --output-dir runs/onshape-gltf
```

Each command emits a digest-bound `geometry.source.json`. The API key is for
local personal or internal automation; it is not service authentication and
must never be installed in Geometry Agent service or Helm configuration.
Continue with:

```bash
content-workflow-cli geometry run \
  --source-manifest runs/onshape-gltf/geometry.source.json \
  --output-dir runs/onshape-geometry
```

Onshape glTF may split B-rep faces into separate open mesh patches. OVRTX and
structural USD checks can still succeed, but do not claim watertight or SimReady
topology unless repair or exact-geometry validation passes. Do not claim image
conditioning or source-history transfer from the MCP.
