"""Explicitly execute ONE prepared native stage; never author an assessment."""

import argparse
import json
import os
from pathlib import Path
import subprocess
from prepare_validation import binding, read, require, sha


def verify_preparation(root, expected_sha):
    receipt_path = root / "preparation_receipt.json"
    require(sha(receipt_path) == expected_sha, "Preparation receipt digest mismatch.")
    receipt = read(receipt_path)
    require(
        receipt["status"] == "PREPARED_NOT_EXECUTED", "Unexpected preparation state."
    )
    require(
        receipt["model"] == "gpt-6-astra" and receipt["reasoning_effort"] == "ultra",
        "Model contract differs.",
    )
    commit = subprocess.check_output(
        ["git", "-C", receipt["repository"], "rev-parse", "HEAD"], text=True
    ).strip()
    require(
        commit == receipt["code_commit"], "Repository commit changed after preparation."
    )
    for item in receipt["input_bindings"] + receipt["prepared_files"]:
        require(
            binding(Path(item["path"])) == item,
            "Prepared input or command changed: " + item["path"],
        )
    return receipt


def launch_environment(repository):
    repo = Path(repository)
    env = os.environ.copy()
    roots = [
        repo,
        repo / "agentic/packages/content_agent_workflows",
        repo / "agentic/packages/content_workflow_cli",
        repo / "apps/usd_cli/src",
    ]
    env["PYTHONPATH"] = os.pathsep.join(map(str, roots)) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--preparation-sha256", required=True)
    p.add_argument(
        "--stage", choices=["run", "collect", "assess", "review"], required=True
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Required to launch; default verifies and prints the command only.",
    )
    a = p.parse_args()
    root = a.prepared.resolve()
    receipt = verify_preparation(root, a.preparation_sha256)
    command = read(root / "commands.json")[a.stage]
    # Refuse known waiver/promotion paths before native typed assessment validation.
    if a.stage == "assess":
        assessment = read(root / "outer_assessment.json")
        require(
            all(
                g.get("disposition") not in {"waive", "defer"}
                for g in assessment.get("gates", [])
            ),
            "Capstone policy forbids waived/deferred gates.",
        )
        require(
            all(
                f.get("disposition") not in {"waived", "deferred"}
                for f in assessment.get("findings", [])
            ),
            "Capstone policy forbids waived/deferred findings.",
        )
    if a.stage == "review":
        require(
            (root / "independent_review.json").is_file(),
            "Independent actual AstraUltra review required.",
        )
    if not a.execute:
        print(
            json.dumps(
                {"stage": a.stage, "executed": False, "command": command}, indent=2
            )
        )
        return
    log = root / (a.stage + ".private.log")
    result_path = root / (a.stage + ".execution.json")
    require(
        not log.exists() and not result_path.exists(),
        "Stage already attempted; retain its outcome and use a fresh preparation for a new attempt.",
    )
    with log.open("xb") as f:
        result = subprocess.run(
            command,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
            cwd=receipt["repository"],
            env=launch_environment(receipt["repository"]),
        )
    unchanged = True
    try:
        verify_preparation(root, a.preparation_sha256)
    except (OSError, ValueError):
        unchanged = False
    record = {
        "stage": a.stage,
        "returncode": result.returncode,
        "inputs_unchanged": unchanged,
        "preparation_sha256": a.preparation_sha256,
        "output_log_sha256": sha(log),
        "terminal_acceptance_inferred_from_exit_code": False,
        "scope": "Native operation result only. Inspect actual assessment/execution/terminal receipt and separate five-seed task conjunction.",
    }
    result_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    raise SystemExit(result.returncode if unchanged else 3)


if __name__ == "__main__":
    main()
