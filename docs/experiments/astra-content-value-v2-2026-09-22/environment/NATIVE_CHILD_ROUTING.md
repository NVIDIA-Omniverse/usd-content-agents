# Native child routing

The unchanged e640 native CLI supports the same named provider configuration
in both arms. The supervisor creates a no-secret config at
`/home/agent/.codex/native_provider.json` using the shared `native_provider.json`
reference. Append these flags to model-backed Physics and Validation commands:

```sh
--codex-config-file /home/agent/.codex/native_provider.json \
  --repo-root /tools/repo --model gpt-6-astra --model-reasoning-effort ultra
```

The named provider uses base URL `http://127.0.0.1:18862/v1`, Responses HTTP,
`env_key=ASTRA_GATEWAY_TOKEN`, no OpenAI account authentication, no websockets
and zero SDK provider retries. Do not use `--codex-base-url`, which selects the
legacy route. No upstream credential is passed to the child.
The supervisor supplies only the run gateway token, including its fresh auth
file. The workflow implementation and model settings are unchanged.

Stage-specific correction: Geometry/general runner supports
`--codex-responses-url` and `--codex-api-key-env` with their
`CONTENT_AGENT_CODEX_RESPONSES_URL` and `CONTENT_AGENT_CODEX_API_KEY_ENV`
defaults. Physics apply and validate run do **not** expose those fields.
Their real parsers and runner configs expose `--codex-config-file` instead.
The earlier overbroad command guidance was caught by the live q11 qualification;
the initial documentation and static constructor receipt are retained under
`environment/history`. The corrected parser-only check executes no handlers.

Use `/tools/repo` as the canonical repository root. `/workflows` exposes the
same read-only code for documentation, but it is a distinct bind alias and does
not match the installed editable package's recorded source path. The native
package verifier correctly rejects that alias. On Linux it also ignores all
ambient Git overrides and global/system config. Environment preparation gives
only the exported repository root and its `.git` directory UID/GID20000, matching
all author lanes; it leaves every file byte, path and mode unchanged and keeps
both namespace mounts read-only. `prepare_native_git_ownership.py` records that
metadata-only change. `probe_native_source_verifier.py` exercises the unchanged
native verifier as UID20000 with the canonical route and the rejected alias.

The bridge uses `--ignore-user-config` and a private child home. Its trusted
configuration escape hatch permits provider fields, not `model_catalog_json`.
The official pinned Codex 0.154.0 builtin catalog already contains `gpt-6-astra`,
supports `ultra`, and specifies `multi_agent_reasoning_effort: xhigh`. Explicit
Ultra is required because the builtin default is low. Safe selected catalog
metadata and the complete upstream-file digest are in
`codex_builtin_astra_metadata.json`; the full catalog/base instructions remain
private. This source finding does not replace actual UI-effort/wire-effort
verification for each native child.

The Python runner stages the actual bridge request in a private
`tempfile.TemporaryDirectory` (`content-workflow-bridge-*` or
`content-workflow-codex-session-*`). The JSON under the run's `raw` directory
is an evidence copy, not the executed request. The JavaScript bridge creates
its launcher and shadow `codex-home/sessions` beside that actual staged request.
With the default temporary directory, these rollouts live in the private
namespace `/tmp`, outside a capture limited to home and workspace. The live q13
qualification exposed this gap; earlier bridge-only path reasoning was incomplete.

A trusted capture must observe those owned namespace temporary files while the
child is live, or both arms must receive an identically configured supported
`TMPDIR` whose scope is captured. The launcher deletes the private home on exit.
A wire request count alone does not establish child UI-effort completeness.
The capture correction needs its own qualification before author launch. Raw
histories remain private. See `native_child_capture_analysis.json` for exact
source bindings; the earlier document remains under `environment/history`.

`check_native_provider_parser.py` tests the actual Physics/Validation parsers and
provider-file loader. `check_native_child_route.mjs` tests the real bridge configuration constructors
without starting the SDK or any model: explicit model/effort, named HTTP route,
key variable name, fixed approval behavior, invalid-route/key rejection, and
exact preservation of the named provider configuration used by Physics/Validation.
`native_child_route_static.json` records this limited PASS. The separate live
native-child qualification remains required before the global freeze.
