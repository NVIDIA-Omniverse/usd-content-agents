# Joint Agent

Use `apps/joint_agent/configs/byoa_joint_rigger.yaml` for public Joint Agent 0.5
Research Preview runs.

```bash
source .venv/bin/activate
export NVIDIA_API_KEY="YOUR_NVIDIA_API_KEY"
export RENDER_ENDPOINT="http://renderer.example:8000"
cp apps/joint_agent/configs/byoa_joint_rigger.yaml my_joint_asset.yaml
# Edit input.usd_path in my_joint_asset.yaml.
joint-agent run my_joint_asset.yaml --dry-run
joint-agent run my_joint_asset.yaml
```

The public template uses `nim`, `qwen/qwen3.5-397b-a17b`, and remote rendering.
Review Stage 2 candidates before enabling `steps.apply_joint_rigger`; the
`owned_core` output is `.joint-agent-byoa/joint_rigger/rigged.usdz`.

Run Gate 3A and Gate 3B with
`.agents/skills/joint-agent-validation/SKILL.md`. Treat their results as static
package/schema evidence, not proof of dynamic simulation behavior. Use
`validation-agent-cli` for separate visual or behavior-evidence checks.

The public path has no external-rigger fallback. Candidate-only compatibility
runs author accepted joint topology but do not author an articulation root.
Prediction-aware owned-core runs author exact aggregate-link membership and
articulation roots when the accepted contract requires them. The path does not
author masses, colliders, drives, joint state, or mimic schemas or prove
simulation readiness. Never print or commit credentials.
