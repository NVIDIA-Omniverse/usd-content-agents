#!/usr/bin/env python3
"""Read-only author submission closure audit; never reads physics results."""
import argparse
from pathlib import Path

from common import closed_file, inventory, object_hash, read, require, sha256, write_new


def inspect(run_root):
    root = Path(run_root)
    reap = read(root / "private/reap.json")
    frozen = read(root / "private/output_inventory.json")
    result = {"schema_version": "author-delivery.v2", "run_id": reap["run_id"],
              "reap_sha256": sha256(root / "private/reap.json"), "claimed_accepted": None,
              "physical_acceptance": "NOT_ASSESSED", "task_outcome": "not_submitted",
              "protocol_validity": "requires independent protocol audit",
              "budget_exhausted": reap["timed_out"], "errors": []}
    try:
        require(reap["cgroup_populated"] == 0 and reap["namespace_init_exited"] is True, "Worker not proven quiescent")
        require(reap["source_unchanged"] is True, "Source changed")
        require(reap["output_snapshot_complete"] is True, "Output closure incomplete")
        require(object_hash(frozen) == reap["output_inventory_sha256"], "Frozen inventory changed")
        require(inventory(root / "workspace", internal_symlinks=True, output_tree=True) == frozen, "Author output changed after reap")
        p = closed_file(root / "workspace", "submission.json")
        require(p.stat().st_size <= 2 * 1024 * 1024, "Oversized submission metadata")
        submission = read(p)
        require(isinstance(submission, dict), "Submission metadata must be a JSON object")
        require(type(submission.get("claimed_accepted")) is bool, "Missing boolean acceptance claim")
        result["claimed_accepted"] = submission["claimed_accepted"]
        require(type(submission.get("repairs_used")) is int and submission["repairs_used"] >= 0, "Invalid repair count")
        require(isinstance(submission.get("reason"), str) and isinstance(submission.get("limitations"), list)
                and all(isinstance(x, str) for x in submission["limitations"]), "Malformed reason/limitations")
        result["declared_repairs_used"] = submission["repairs_used"]
        result["submission_sha256"] = sha256(p)
        scene, bindings = submission.get("final_scene"), submission.get("bindings")
        require((scene is None) == (bindings is None), "Scene and bindings must both be present or both null")
        if scene is None:
            require(submission["claimed_accepted"] is False, "Cannot claim acceptance without a scene")
            result["task_outcome"] = "not_submitted"
        else:
            scene_file = closed_file(root / "workspace", scene)
            bindings_file = closed_file(root / "workspace", bindings)
            result.update(task_outcome="submitted", final_scene=scene, bindings=bindings,
                          final_scene_sha256=sha256(scene_file), bindings_sha256=sha256(bindings_file))
        result["artifact_closure_valid"] = True
    except (OSError, ValueError, KeyError) as exc:
        result["artifact_closure_valid"] = False
        result["task_outcome"] = "invalid_or_missing_submission"
        result["errors"].append(type(exc).__name__ + ": " + str(exc))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--run-root", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); write_new(a.output, inspect(a.run_root))
