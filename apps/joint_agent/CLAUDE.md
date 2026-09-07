# Joint Agent

Use `apps/joint_agent/configs/byoa_joint_rigger.yaml` for public Joint Agent 0.5
Research Preview runs.

This guide applies only after the user explicitly selects the fixed-pipeline
Joint Agent. Route an unqualified articulation task through
`content-workflow-cli articulation run` from the repository root.

The fixed-pipeline Joint Agent supports Linux and Windows. On Windows, run it
under WSL2 with the `warp` rendering backend; native Windows execution and
local OVRTX rendering under WSL2 are not supported.

```bash
source .venv/bin/activate
uv pip install -e ".[warp]" -e apps/joint_agent
export NVIDIA_API_KEY="YOUR_NVIDIA_API_KEY"
export RENDER_ENDPOINT="http://renderer.example:8000"
cp apps/joint_agent/configs/byoa_joint_rigger.yaml my_joint_asset.yaml
# Edit input.usd_path in my_joint_asset.yaml.
joint-agent run my_joint_asset.yaml --dry-run
joint-agent run my_joint_asset.yaml
```

The public template uses `nim`, `moonshotai/kimi-k3`, and remote rendering
for Linux runs. On WSL2, the root `warp` extra above supplies the required
Warp/Newton runtime; set both `steps.identify_asset.renderer.backend` and
`steps.build_dataset_usd.renderer.backend` to `warp` as required by the support
policy above. Review Stage 2 candidates before enabling
`steps.apply_joint_rigger`; the `owned_core` output is
`.joint-agent-byoa/joint_rigger/rigged.usdz`.

Run Gate 3A and Gate 3B with
`.agents/skills/fixed-pipeline/references/joint-agent-validation/reference.md`. Treat their results as static
package/schema evidence, not proof of dynamic simulation behavior. Use
`validation-agent-cli` for separate visual or behavior-evidence checks.

The public path has no external-rigger fallback. Candidate-only compatibility
runs author accepted joint topology but do not author an articulation root.
Prediction-aware owned-core runs author exact aggregate-link membership and
articulation roots when the accepted contract requires them. The path does not
author masses, colliders, drives, joint state, or mimic schemas or prove
simulation readiness. Never print or commit credentials.
