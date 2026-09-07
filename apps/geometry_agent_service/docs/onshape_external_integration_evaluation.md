# Onshape Authoring and Export Evaluation

## Product boundary

Onshape authoring uses the official Onshape Labs FeatureScript MCP at
`https://fs-mcp.labs.onshape.app/mcp`. The user subscribes to the application
and completes Onshape authentication in their MCP client. The MCP creates and
tests parametric FeatureScript geometry in Onshape.

The evaluated MCP tool surface did not include geometry export or download.
After authoring, either:

1. the user exports a supported file from Onshape and submits it through the
   ordinary existing-asset path; or
2. the user configures an Onshape API key in the local execution environment and
   runs `geometry-agent export-onshape`.

The second route is export-only. It cannot author or revise geometry. It freezes
a mutable workspace as a named immutable version before export, or accepts an
existing version ID. It returns a bounded, digest-verified
`geometry.source.v1` package containing geometry, units, axes, immutable source
identity, and provenance.

The deployed Geometry Agent service never receives the user's Onshape OAuth
session, API key, or secret. There are no Onshape credential settings in its
service configuration or Helm chart.

The public 0.6 Geometry package and service image also do not ship a CAD/mesh
converter. STEP, glTF, and OBJ exports remain immutable source handoffs. Running
validation or OVRTX evidence requires an existing or provider-produced
USD-family asset, or a separately qualified converter deployment.

## Secure local export

Configure `ONSHAPE_API_KEY` and `ONSHAPE_API_SECRET` through a local secret
manager or process environment outside the agent conversation. The compatible
legacy environment name `ONSHAPE_SECRET` is also accepted. Do not paste either
credential into chat, put it in command arguments, enable shell tracing, or
write it to a request file.

Export a mutable workspace safely:

```bash
geometry-agent export-onshape \
  --document-id DOCUMENT_ID \
  --workspace-id WORKSPACE_ID \
  --snapshot-name "Geometry handoff" \
  --element-id ELEMENT_ID \
  --element-kind partstudio \
  --format gltf \
  --rights-assertion "Authorized to export this Onshape element." \
  --output-dir runs/onshape-source
```

For an existing immutable version, replace `--workspace-id` and
`--snapshot-name` with `--version-id VERSION_ID`.

The connector signs each request independently, disables redirects and ambient
proxy credentials, bounds timeouts and response sizes, validates archive paths
and dependencies, and strips provider response details from public errors. It
uses only official `onshape.com` API hosts. The command has no credential flags.

## Live evaluation

Evaluation date: 2026-08-26.

### Authoring

The authenticated MCP reported FeatureScript library version `3044`. Two
parameterized features were tested and created:

| Object | Semantic controls | Result |
| --- | --- | --- |
| Keyboard | rows, columns, horizontal and vertical pitch, key width/depth/height, base thickness, side margin | compiled and evaluated successfully; no notices; top/front/left images returned |
| Duplex outlet | plate width/height/thickness, receptacle spacing/radius/rise | compiled and evaluated successfully; no notices; top/front/left images returned |

The first keyboard draft also exercised the fail-correct loop. Onshape rejected
an assumed bound symbol; the feature was corrected to use an explicit
`LengthBoundSpec`, retested, and created only after all errors were cleared.

### Export and Geometry handoff

The keyboard workspace was exported live with a locally configured API key:

| Check | Result |
| --- | --- |
| Mutable-workspace protection | pass; a named immutable version was created before export |
| Exact CAD artifact | pass; STEP was downloaded and digest-bound |
| Render representation | pass; glTF was downloaded and digest-bound |
| Credential scan | pass; neither credential occurred in output files or manifests |
| Coordinate handoff | pass after fixing glTF to declare Y-up and millimeter units |
| USD conversion and structure | pass |
| OVRTX six-view render and framing audit | pass |
| Mesh watertightness | not certified; the glTF represented B-rep faces as separate open mesh patches |

The STEP file remains the exact CAD exchange artifact. In the historical
evaluation environment, the glTF representation continued through USD
conversion and visual evidence when exact B-rep conversion was unavailable. In
the public 0.6 service, it is retained and the downstream run fails closed until
a qualified converter or provider-produced USD is available. A visual route
does not claim that the mesh is watertight or fully simulation-ready.

## Agent procedure

1. Connect the user's MCP client to the official URL and let the user authorize.
2. Discover the current tools and read the persistent FeatureScript notes.
3. Query the live FeatureScript library version instead of assuming one.
4. Write parameterized FeatureScript with defaults for every exposed parameter.
5. Test the feature and fix every notice or error before creating geometry.
6. Inspect the current MCP tools. If export is unavailable, offer manual export
   first.
7. If automatic export is needed, ask the user to configure
   `ONSHAPE_API_KEY` and `ONSHAPE_API_SECRET` in the local execution environment
   or secret manager. Ask the user only to confirm configuration; never ask them
   to paste either value.
8. Snapshot a mutable workspace before exporting, then submit the resulting
   source manifest or exported file to Geometry Agent.
9. Preserve STEP for exact CAD exchange and glTF as a render-oriented source
   handoff. Use glTF for evidence only through a separately qualified converter
   deployment, and report topology limits explicitly.

Do not claim that the MCP exports files, that the export helper authors
geometry, or that a renderable mesh is automatically watertight or
simulation-ready.

## Official references

- <https://www.onshape.com/en/features/onshape-labs>
- <https://onshape-public.github.io/docs/auth/apikeys/>
- <https://onshape-public.github.io/docs/api-adv/translation/>
- <https://onshape-public.github.io/docs/api-adv/documents/>
