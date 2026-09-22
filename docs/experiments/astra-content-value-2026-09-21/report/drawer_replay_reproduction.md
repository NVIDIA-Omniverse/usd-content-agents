# Independent drawer replay presentation

The three snapshots preserve the observed drawer and free payload poses from seed 11 of the frozen independent evaluator. They illustrate the settling, open-and-loaded, and returned-closed phases. Acceptance comes from all five native physical trials in `five_seed_summary.json`, not image appearance. Camera, lighting, and payload display color are presentation additions. The original scored output, evaluator scene, trace, and replay are unchanged.

The snapshot directory is portable: keep its three USDA files, `assets/`, and `replay_render_config.json` together. The renderer resolves snapshot filenames relative to this directory and verifies the saved scene and texture hashes. It does not load author code or rerun physics. Use Linux, an existing qualified USD CLI runtime, OVRTX `0.4.1.364340`, a Python environment with NumPy and Pillow, and an exclusively allocated GPU UUID. From the experiment directory, copy just the inputs into a new directory; the published directory already contains completed render receipts and the script deliberately refuses to overwrite them:

```sh
export SNAPSHOT_REPLAY=/absolute/path/to/new-drawer-render
export REPLAY_GPU_UUID=GPU-your-allocated-device-uuid
mkdir "$SNAPSHOT_REPLAY"
for name in closed_initial open_loaded closed_returned; do
  cp "visuals/pilot-v1/01_drawer/plain_astra/$name.usda" "$SNAPSHOT_REPLAY/"
done
cp visuals/pilot-v1/01_drawer/plain_astra/replay_render_config.json "$SNAPSHOT_REPLAY/"
cp -R visuals/pilot-v1/01_drawer/plain_astra/assets "$SNAPSHOT_REPLAY/assets"
nice -n15 ionice -c3 /path/to/runtime/python scripts/render_portable_drawer_snapshots.py \
  --snapshots "$SNAPSHOT_REPLAY" \
  --usd-cli /path/to/repo/.venv/bin/usd-cli \
  --ovrtx-venv /path/to/ovrtx-venv \
  --gpu-uuid "$REPLAY_GPU_UUID"
```

The script uses its own working directory, CLI configuration, and session, renders 1024x768 through OVRTX, records per-command elapsed times and image hashes, and stops only its own CLI server. Existing render attempts are not overwritten; copy the immutable snapshot inputs into a new directory for a repeat. The runtime used for the recorded attempt is documented in `render_provenance.json` and the native render receipts. Installation/provisioning is separate from this presentation script.

The recorded run used OVRTX `rt2` with 64 sensor updates and completed the three frames in 26.424 seconds. These are recorded loaded-host timings, not a portable performance promise. The first attempt failed before rendering because the isolated CLI configuration omitted `server.allowed_roots`; its diagnostic is retained under `attempt01_open_policy_failure/`. The current portable script sets that permission to the snapshot directory. No scene or physics correction was made. Public JSON projections remove runtime identifiers; original retained receipt hashes remain identified in `publication_manifest.json`, and should not be mistaken for hashes of the projected JSON bytes.

The original accepted `final.usd` also relocates without edits when `bindings.json` and `assets/textures/` stay beside it. `drawer_portability_receipt.json` records an independent relocation and USD dependency readback: all three textures resolved inside the new directory and zero dependencies were unresolved. `scripts/check_drawer_portability.py` reproduces that check without changing the scored bytes.
