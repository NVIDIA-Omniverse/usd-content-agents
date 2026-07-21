# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Physics Agent REST client skill contract."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = REPO_ROOT / ".agents" / "skills" / "physics-agent-client" / "SKILL.md"
HELPER_PATH = SKILL_PATH.parent / "scripts" / "request_helpers.sh"


def _skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def _helper_text() -> str:
    return HELPER_PATH.read_text(encoding="utf-8")


def _section(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def _bash_block(section: str) -> str:
    return section.split("```bash", 1)[1].split("```", 1)[0]


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _run_helper(
    tmp_path: Path,
    *,
    curl_script: str,
    command: str,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    curl_path = tmp_path / "curl"
    curl_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + curl_script,
        encoding="utf-8",
    )
    curl_path.chmod(0o755)
    env = os.environ.copy()
    env.update(env_overrides or {})
    env["HELPER_PATH"] = str(HELPER_PATH)
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    return subprocess.run(
        ("bash", "-c", f'source "$HELPER_PATH"\n{command}'),
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=env,
    )


def test_client_request_helpers_are_executable_valid_bash() -> None:
    assert os.access(HELPER_PATH, os.X_OK)
    subprocess.run(("bash", "-n", str(HELPER_PATH)), check=True)


def test_client_skill_stops_at_unreachable_service_boundary() -> None:
    text = _normalized(_skill_text())

    assert "If `GET /health` is unreachable, stop client execution" in text
    assert "Do not build, start, restart, or reconfigure the service" in text
    assert "`deploy-physics-agent-docker`" in text
    assert "`deploy-physics-agent-brev`" in text
    assert "Start the Physics Agent service or correct `BASE_URL`" not in text


def test_client_skill_documents_route_specific_input_contracts() -> None:
    text = _skill_text()
    pipeline = _section(text, "## Key Pipeline Parameters", "## Key Predict Parameters")
    predict = _section(text, "## Key Predict Parameters", "## Key Tune Parameters")
    tune = _section(text, "## Key Tune Parameters", "## Key Refine Parameters")
    refine = _section(text, "## Key Refine Parameters", "## Output Format")

    normalized_pipeline = _normalized(pipeline)
    assert (
        "At least one of `usd_file`, `session_id`, or `s3_uri`" in normalized_pipeline
    )
    assert "selects `session_id`, then `s3_uri`, then `usd_file`" in normalized_pipeline

    for form_name in ("`usd_file`", "`session_id`", "`s3_uri`", "`dataset_path`"):
        assert form_name in predict
    assert "`dataset_path` may be used alone or with `session_id`" in _normalized(
        predict
    )

    for form_name in (
        "`physics_usd`",
        "`source_session_id`",
        "`s3_uri`",
        "`scenario_yaml`",
        "`user_prompt`",
    ):
        assert form_name in tune
        assert form_name in refine
    assert "not `usd_file`" in tune


def test_client_skill_curl_uses_route_specific_upload_fields() -> None:
    text = _skill_text()
    predict = _section(text, "### Predict Workflow", "### Tune Workflow")
    tune = _section(text, "### Tune Workflow", "### Refine Workflow")
    predict_bash = _bash_block(predict)
    tune_bash = _bash_block(tune)

    assert '-F "usd_file=@scene.usd"' in predict_bash
    assert '-F "physics_usd=@physics.usd"' in tune_bash
    assert "usd_file" not in tune_bash


def test_client_skill_health_preflight_hard_stops_before_submission() -> None:
    skill = _normalized(_skill_text())
    curl_workflow = _section(_skill_text(), "## curl Workflow", "Optimizer examples:")
    bash = _bash_block(curl_workflow)
    helper = _helper_text()

    assert "set -euo pipefail" in bash
    assert "Run the curl workflow's `physics_client_health_preflight`" in skill
    assert "Treat its nonzero result as a hard stop" in skill
    assert (
        "source .agents/skills/physics-agent-client/scripts/request_helpers.sh" in bash
    )
    health_failure_branch = "if ! physics_client_health_preflight; then\n  exit 1\nfi"
    assert health_failure_branch in bash
    assert bash.index(health_failure_branch) < bash.index(
        'PIPELINE_RESPONSE=$(physics_client_submit "/pipeline"'
    )
    assert "--connect-timeout" in helper
    assert "--max-time" in helper
    assert '"$BASE_URL/health"' in helper
    assert "if ! jq -e ." in helper


def test_health_preflight_rejects_malformed_json_with_handoff(
    tmp_path: Path,
) -> None:
    args_log = tmp_path / "curl-args"
    result = _run_helper(
        tmp_path,
        curl_script="printf '%s\\n' \"$@\" >\"$CURL_ARGS_LOG\"\nprintf 'not-json\\n'\n",
        command='export BASE_URL="http://service"\nphysics_client_health_preflight',
        env_overrides={"CURL_ARGS_LOG": str(args_log)},
    )

    assert result.returncode == 1
    assert "stop client execution and hand deployment" in result.stderr
    curl_args = args_log.read_text(encoding="utf-8")
    assert "--connect-timeout\n10\n" in curl_args
    assert "--max-time\n600\n" in curl_args


def test_submission_aborts_before_post_when_ledger_cannot_be_created(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "curl-invoked"
    result = _run_helper(
        tmp_path,
        curl_script='printf invoked >"$CURL_MARKER"\n',
        command=(
            'export BASE_URL="http://service" REQUEST_LEDGER="$BAD_LEDGER"\n'
            'physics_client_submit "/pipeline" "usd_file" -X POST '
            '"$BASE_URL/pipeline"'
        ),
        env_overrides={
            "BAD_LEDGER": str(tmp_path / "missing" / "ledger.jsonl"),
            "CURL_MARKER": str(marker),
        },
    )

    assert result.returncode == 1
    assert "Cannot create or write request ledger" in result.stderr
    assert not marker.exists()


def test_submission_rejects_curl_fail_mode_before_post(tmp_path: Path) -> None:
    marker = tmp_path / "curl-invoked"
    result = _run_helper(
        tmp_path,
        curl_script='printf invoked >"$CURL_MARKER"\n',
        command=(
            'export BASE_URL="http://service" REQUEST_LEDGER="$LEDGER"\n'
            'physics_client_submit "/pipeline" "usd_file" -fsS -X POST '
            '"$BASE_URL/pipeline"'
        ),
        env_overrides={
            "CURL_MARKER": str(marker),
            "LEDGER": str(tmp_path / "ledger.jsonl"),
        },
    )

    assert result.returncode == 2
    assert "rejects curl fail mode" in result.stderr
    assert not marker.exists()


def test_submission_preserves_http_status_before_returning_curl_failure(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    result = _run_helper(
        tmp_path,
        curl_script="""
response_file=""
while (($#)); do
  case "$1" in
    -o) response_file="$2"; shift 2 ;;
    *) shift ;;
  esac
done
printf '{"detail":"too large"}' >"$response_file"
printf '413'
printf 'curl: HTTP failure\\n' >&2
exit 22
""",
        command=(
            'export BASE_URL="http://service" REQUEST_LEDGER="$LEDGER"\n'
            'physics_client_submit "/pipeline" "usd_file" --fail-with-body '
            '-X POST "$BASE_URL/pipeline"'
        ),
        env_overrides={"LEDGER": str(ledger)},
    )

    assert result.returncode == 22
    entry = json.loads(ledger.read_text(encoding="utf-8"))
    assert entry["http_status"] == "413"
    assert entry["transport_exit_code"] == 22
    assert entry["response"] == '{"detail":"too large"}'
    assert entry["transport_error"] is None


def test_submission_records_accepted_response_before_returning_success(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    result = _run_helper(
        tmp_path,
        curl_script="""
response_file=""
while (($#)); do
  case "$1" in
    -o) response_file="$2"; shift 2 ;;
    *) shift ;;
  esac
done
printf '{"session_id":"session-1"}' >"$response_file"
printf '202'
""",
        command=(
            'export BASE_URL="http://service" REQUEST_LEDGER="$LEDGER"\n'
            'physics_client_submit "/pipeline" "usd_file" -X POST '
            '"$BASE_URL/pipeline"'
        ),
        env_overrides={"LEDGER": str(ledger)},
    )

    assert result.returncode == 0
    assert result.stdout == '{"session_id":"session-1"}'
    entry = json.loads(ledger.read_text(encoding="utf-8"))
    assert entry["http_status"] == "202"
    assert entry["transport_exit_code"] == 0
    assert entry["session_id"] == "session-1"


def test_submission_returns_failure_when_post_response_cannot_be_ledgered(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    marker = tmp_path / "curl-invoked"
    result = _run_helper(
        tmp_path,
        curl_script="""
response_file=""
while (($#)); do
  case "$1" in
    -o) response_file="$2"; shift 2 ;;
    *) shift ;;
  esac
done
printf invoked >"$CURL_MARKER"
rm -f "$REQUEST_LEDGER"
mkdir "$REQUEST_LEDGER"
printf '{"session_id":"accepted-session"}' >"$response_file"
printf '201'
""",
        command=(
            'export BASE_URL="http://service" REQUEST_LEDGER="$LEDGER"\n'
            'physics_client_submit "/pipeline" "usd_file" -X POST '
            '"$BASE_URL/pipeline"'
        ),
        env_overrides={"LEDGER": str(ledger), "CURL_MARKER": str(marker)},
    )

    assert marker.exists()
    assert result.returncode == 1
    assert result.stdout == ""
    assert "Cannot append request ledger" in result.stderr


def test_polling_has_a_finite_terminal_status_deadline(tmp_path: Path) -> None:
    result = _run_helper(
        tmp_path,
        curl_script='printf \'{"status":"running"}\\n\'\n',
        command=(
            'export BASE_URL="http://service" '
            "PHYSICS_CLIENT_CONNECT_TIMEOUT_S=1 "
            "PHYSICS_CLIENT_REQUEST_TIMEOUT_S=1 "
            "PHYSICS_CLIENT_POLL_TIMEOUT_S=1 "
            "PHYSICS_CLIENT_POLL_INTERVAL_S=1\n"
            "physics_client_poll_until_terminal pipeline session-1"
        ),
    )

    assert result.returncode == 1
    assert "Timed out waiting for pipeline session session-1" in result.stderr


def test_client_skill_ledgers_responses_before_submission_errors_escape() -> None:
    text = _skill_text()
    accounting = _normalized(
        _section(text, "### Submission Accounting", "## Python Client")
    )
    helper = _helper_text()

    assert "Record an attempt before errors escape the submission call." in accounting
    assert "HTTPError.response.status_code" in accounting
    assert "HTTPError.response.text" in accounting
    assert "Do not count `run_and_monitor` as one call" in accounting
    for field in (
        "attempt",
        "endpoint",
        "input_mode",
        "http_status",
        "response",
        "session_id",
    ):
        assert field in helper
    append_at = helper.index("if ! printf '%s\\n' \"$ledger_entry\"")
    assert 'if ! touch "$REQUEST_LEDGER"' in helper
    assert 'if ! : >>"$REQUEST_LEDGER"' in helper
    assert append_at < helper.index('return "$curl_exit"')
    assert append_at < helper.index("return 22")


def test_client_skill_polls_async_routes_before_requesting_results() -> None:
    text = _skill_text()
    for start, end, family in (
        ("### Predict Workflow", "### Tune Workflow", "predict"),
        ("### Tune Workflow", "### Refine Workflow", "tune"),
        ("### Refine Workflow", "## Endpoint Reference", "refine"),
    ):
        bash = _bash_block(_section(text, start, end))
        poll = f"physics_client_poll_until_terminal {family}"
        results = f'"$BASE_URL/{family}/$'
        assert poll in bash
        assert '!= "completed"' in bash
        assert bash.index(poll) < bash.index(results)


def test_client_skill_polls_optimizer_examples_to_terminal_status() -> None:
    optimizer = _bash_block(
        _section(_skill_text(), "Optimizer examples:", "### Predict Workflow")
    )

    for prefix in ("OPTIMIZED", "SPLIT"):
        assert f"{prefix}_SESSION=$(jq -er '.session_id'" in optimizer
        assert (
            f"{prefix}_STATUS=$(physics_client_poll_until_terminal pipeline"
            in optimizer
        )
        assert f'[[ "${prefix}_STATUS" == "completed" ]] || exit 1' in optimizer


def test_client_skill_requires_exact_attempt_and_session_accounting() -> None:
    output = _normalized(
        _section(_skill_text(), "## Output Format", "## Troubleshooting")
    )

    assert "one entry per `POST`" in output
    assert "classify every `POST` attempt once by transport outcome" in output
    assert "accepted responses (`2xx`)" in output
    assert "rejected pre-session requests" in output
    assert "server failures (`5xx`)" in output
    assert "unexpected HTTP responses (`1xx` or `3xx`)" in output
    assert "transport failures (no HTTP status)" in output
    assert "Record session-ID presence separately" in output
    assert "unexpected presence or absence is a contract anomaly" in output
    assert (
        "execution attempts keyed by `(route family, session ID, ledger attempt or "
        "generation)`" in output
    )
    assert "successful (`completed`), `failed`, and `cancelled`" in output
    assert "`/pipeline/upload-usd` is an accepted upload-only request" in output
    assert "Retries and `/regenerate` calls count separately" in output
    assert "capture each terminal status before it is overwritten" in output
    assert "not a failed session" in output


def test_client_skill_documents_all_predict_preparation_controls() -> None:
    predict = _section(
        _skill_text(), "## Key Predict Parameters", "## Key Tune Parameters"
    )

    for field in (
        "`render_backend`",
        "`enable_deinstance`",
        "`enable_split`",
        "`enable_deduplicate`",
    ):
        assert field in predict


def test_python_examples_use_bounded_polling_and_explicit_post_helpers() -> None:
    python = _section(_skill_text(), "## Python Client", "## curl Workflow")

    assert "time.monotonic() + timeout_s" in python
    assert "raise TimeoutError" in python
    assert "client.run_and_monitor(" not in python
    assert "client.upload_usd(" in python
    assert "client.start_pipeline(" in python


def test_documented_direct_curl_calls_have_finite_timeouts() -> None:
    curl_workflow = _section(_skill_text(), "## curl Workflow", "## Endpoint Reference")

    assert "PHYSICS_CLIENT_CONNECT_TIMEOUT_S=10" in curl_workflow
    assert "PHYSICS_CLIENT_REQUEST_TIMEOUT_S=600" in curl_workflow
    for line in curl_workflow.splitlines():
        if line.lstrip().startswith("curl "):
            assert '"${CURL_TIMEOUTS[@]}"' in line


def test_client_skill_keeps_troubleshooting_remedies_evidence_bound() -> None:
    text = _skill_text()
    troubleshooting = text.split("## Troubleshooting", 1)[1]

    assert "Missing `physical_properties`, invalid VLM output/schema" in troubleshooting
    assert (
        "Do not enable optimizer flags unless separate geometry evidence"
        in troubleshooting
    )
    assert "Error literally identifies an instance proxy" in troubleshooting
    assert "Add `enable_split=true` only when" in troubleshooting


def test_client_skill_disqualifies_fake_engine_release_evidence() -> None:
    text = _normalized(_skill_text())

    assert (
        "A synthetic `fake` engine run is test infrastructure only. It is never "
        "production, release, simulator, tuning-quality, or acceptance evidence."
        in text
    )


def test_client_skill_routes_upload_limit_changes_outside_client_scope() -> None:
    troubleshooting = _normalized(_skill_text().split("## Troubleshooting", 1)[1])

    assert "Increase `PA_MAX_UPLOAD_SIZE_MB`" not in troubleshooting
    assert "Use a smaller file or submit an S3 URI." in troubleshooting
    assert "`deploy-physics-agent-docker`" in troubleshooting
    assert "`deploy-physics-agent-brev`" in troubleshooting
    assert "do not reconfigure it from this client skill" in troubleshooting
