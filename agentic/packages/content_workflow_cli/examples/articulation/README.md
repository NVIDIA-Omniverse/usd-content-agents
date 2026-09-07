# Reviewed Articulation Workflow

From the repository root, run the practical reviewed-articulation scenario with
a copied, configured Joint Agent YAML:

```bash
ARTICULATION_INTENT="Identify the six drawers, present every candidate for "
ARTICULATION_INTENT+="review, then author exactly six prismatic drawer joints "
ARTICULATION_INTENT+="and no masses or colliders."

content-workflow-cli articulation run \
  --usd path/to/file_cabinet.usdz \
  --joint-config path/to/joint-agent.yaml \
  --output-dir runs/file-cabinet-articulation \
  --intent "$ARTICULATION_INTENT" \
  --review-policy all \
  --allowed-motion-type prismatic \
  --expected-candidate-count 6 \
  --max-candidate-count 6
```

A `needs_review` result is a successful pause. Inspect
`articulation_candidates.json` and `scene_evidence/manifest.json`, then
write one `accept` or `reject` decision for every printed review-required
candidate ID:

```json
{
  "candidate_0001": "accept",
  "candidate_0002": "accept",
  "candidate_0003": "accept",
  "candidate_0004": "accept",
  "candidate_0005": "accept",
  "candidate_0006": "accept"
}
```

Use the exact IDs printed by the run.

Bind the receipt and resume:

```bash
content-workflow-cli articulation review \
  --run-dir runs/file-cabinet-articulation \
  --decisions-json path/to/decisions.json \
  --reviewer asset-owner
```

After an interruption, run:

```bash
content-workflow-cli articulation resume \
  --run-dir runs/file-cabinet-articulation
```

Repeating the exact `run` command also resumes without repeating completed
inference, `usd-cli` evidence, or authoring work.

## Provider-neutral standalone route

For a run that must not construct the classic Joint client, first use
`content-articulation-inspection` and `articulation publish-preparation` to seal
the exact source, dependency, hierarchy, membership, owner, capability, Scene,
and render evidence. Validate the publication, then pass its preparation to the
same public launcher:

```bash
content-workflow-cli articulation validate-preparation \
  --publication runs/articulation-preparation/articulation_preparation_publication.json

content-workflow-cli articulation run \
  --usd path/to/file_cabinet.usdz \
  --preparation runs/articulation-preparation/embedded_articulation_preparation.json \
  --output-dir runs/file-cabinet-standalone \
  --intent "$ARTICULATION_INTENT" \
  --allowed-motion-type prismatic \
  --expected-candidate-count 6 \
  --max-candidate-count 6
```

The single child is confined to the semantic decision patch. Ledger/graph
publication, accepted-only graph-authoring, saved-stage readback, canonical
OVRTX evidence validation, and terminal publication remain deterministic outer
operations. A distinct outer coordinator must inspect the retained OVRTX image
digests and write the post-review patch; the workflow does not launch another
child automatically. A conditional result preserves the candidate-bound issue
packet and one immutable refinement attempt; it is not a successful
articulation outcome.
