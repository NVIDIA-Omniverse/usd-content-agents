# Joint Agent

Joint Agent 0.5 is a Research Preview for articulated-component classification,
Stage 2 candidate inference, and topology-only USDZ publication.

## Quickstart

Run from the repository root:

```bash
source .venv/bin/activate
export NVIDIA_API_KEY="YOUR_NVIDIA_API_KEY"
export RENDER_ENDPOINT="http://renderer.example:8000"
cp apps/joint_agent/configs/byoa_joint_rigger.yaml my_joint_asset.yaml
# Edit input.usd_path in my_joint_asset.yaml.
joint-agent run my_joint_asset.yaml --dry-run
joint-agent run my_joint_asset.yaml
```

The template uses `nim`, `qwen/qwen3.5-397b-a17b`, and remote rendering. Review
the generated Stage 2 candidate document before setting
`steps.apply_joint_rigger.enabled: true`. Resume the run to publish
`.joint-agent-byoa/joint_rigger/rigged.usdz` through `owned_core`.

## Validation

Follow `.agents/skills/joint-agent-validation/SKILL.md` to run Gate 3A and Gate
3B on the final USDZ. These are static checks and do not prove dynamic behavior.
Use `validation-agent-cli` for separate visual or behavior-evidence checks.

## Limits

The public authoring path is topology-only and has no external-rigger fallback.
Candidate-only compatibility runs author accepted joint topology but do not
author an articulation root. Prediction-aware owned-core runs author exact
aggregate-link membership and articulation roots when the accepted contract
requires them.
The path does not add masses, colliders, drives, joint state, or mimic schemas,
and it does not prove simulation readiness. Keep credentials in local
environment files and do not commit them.
