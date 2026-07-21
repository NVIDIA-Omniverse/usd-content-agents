#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

physics_client_health_preflight() {
  : "${BASE_URL:?Set BASE_URL before calling physics_client_health_preflight}"

  local response
  local connect_timeout="${PHYSICS_CLIENT_CONNECT_TIMEOUT_S:-10}"
  local request_timeout="${PHYSICS_CLIENT_REQUEST_TIMEOUT_S:-600}"
  local handoff_message="Health check failed; stop client execution and hand deployment to deploy-physics-agent-docker or deploy-physics-agent-brev."
  if ! response=$(curl --connect-timeout "$connect_timeout" \
    --max-time "$request_timeout" -fsS "$BASE_URL/health"); then
    printf '%s\n' "$handoff_message" >&2
    return 1
  fi
  if ! jq -e . <<<"$response" >/dev/null 2>&1; then
    printf '%s\n' "$handoff_message" >&2
    return 1
  fi
  jq . <<<"$response"
}

physics_client_submit() {
  : "${REQUEST_LEDGER:?Set REQUEST_LEDGER before calling physics_client_submit}"

  local endpoint="$1" input_mode="$2"
  shift 2

  local response_file error_file http_status curl_exit=0 response error_text
  local session_id attempt ledger_entry observed_http_status arg
  local connect_timeout="${PHYSICS_CLIENT_CONNECT_TIMEOUT_S:-10}"
  local request_timeout="${PHYSICS_CLIENT_REQUEST_TIMEOUT_S:-600}"

  for arg in "$@"; do
    if [[ "$arg" == "--fail" || ("$arg" != --* && "$arg" == -*f*) ]]; then
      printf '%s\n' \
        "physics_client_submit rejects curl fail mode; let it capture the HTTP body and status before returning nonzero." >&2
      return 2
    fi
  done
  if [[ -e "$REQUEST_LEDGER" && ! -f "$REQUEST_LEDGER" ]]; then
    printf 'Request ledger is not a regular file: %s\n' "$REQUEST_LEDGER" >&2
    return 1
  fi
  if ! touch "$REQUEST_LEDGER"; then
    printf 'Cannot create or write request ledger: %s\n' "$REQUEST_LEDGER" >&2
    return 1
  fi
  if ! : >>"$REQUEST_LEDGER"; then
    printf 'Cannot append request ledger: %s\n' "$REQUEST_LEDGER" >&2
    return 1
  fi
  if ! attempt=$(wc -l <"$REQUEST_LEDGER"); then
    printf 'Cannot read request ledger: %s\n' "$REQUEST_LEDGER" >&2
    return 1
  fi
  attempt=$((attempt + 1))
  if ! response_file=$(mktemp); then
    printf '%s\n' "Cannot allocate response capture file; submission aborted." >&2
    return 1
  fi
  if ! error_file=$(mktemp); then
    rm -f "$response_file"
    printf '%s\n' "Cannot allocate error capture file; submission aborted." >&2
    return 1
  fi

  if http_status=$(curl -sS -o "$response_file" -w '%{http_code}' "$@" \
    --connect-timeout "$connect_timeout" --max-time "$request_timeout" \
    2>"$error_file"); then
    curl_exit=0
  else
    curl_exit=$?
  fi

  response=$(<"$response_file")
  error_text=$(<"$error_file")
  session_id=$(jq -r '.session_id // empty' "$response_file" 2>/dev/null || true)
  if [[ "$http_status" =~ ^[0-9]{3}$ && "$http_status" != "000" ]]; then
    observed_http_status="$http_status"
  else
    observed_http_status="transport_error"
  fi

  if ! ledger_entry=$(jq -cn \
    --argjson attempt "$attempt" \
    --arg endpoint "$endpoint" \
    --arg input_mode "$input_mode" \
    --arg http_status "$observed_http_status" \
    --argjson transport_exit_code "$curl_exit" \
    --arg response "$response" \
    --arg transport_error "$error_text" \
    --arg session_id "$session_id" \
    '{attempt: $attempt, endpoint: $endpoint, input_mode: $input_mode,
      http_status: $http_status, transport_exit_code: $transport_exit_code,
      response: $response,
      transport_error: (if $http_status == "transport_error" and $transport_error != ""
        then $transport_error else null end),
      session_id: (if $session_id == "" then null else $session_id end)}'); then
    rm -f "$response_file" "$error_file"
    printf '%s\n' "Cannot serialize request ledger entry; submission cannot be accounted for." >&2
    return 1
  fi
  if ! printf '%s\n' "$ledger_entry" >>"$REQUEST_LEDGER"; then
    rm -f "$response_file" "$error_file"
    printf 'Cannot append request ledger: %s\n' "$REQUEST_LEDGER" >&2
    return 1
  fi

  cat "$response_file"
  rm -f "$response_file" "$error_file"
  if [[ "$observed_http_status" != "transport_error" && \
    ! "$observed_http_status" =~ ^2[0-9][0-9]$ ]]; then
    printf '%s\n' "$response" >&2
    return 22
  fi
  if ((curl_exit != 0)); then
    printf '%s\n' "$error_text" >&2
    return "$curl_exit"
  fi
  if [[ "$observed_http_status" == "transport_error" ]]; then
    printf '%s\n' "$error_text" >&2
    return 1
  fi
}

physics_client_poll_until_terminal() {
  : "${BASE_URL:?Set BASE_URL before calling physics_client_poll_until_terminal}"

  local family="$1" session_id="$2" status_response status
  local connect_timeout="${PHYSICS_CLIENT_CONNECT_TIMEOUT_S:-10}"
  local request_timeout="${PHYSICS_CLIENT_REQUEST_TIMEOUT_S:-600}"
  local poll_timeout="${PHYSICS_CLIENT_POLL_TIMEOUT_S:-1800}"
  local poll_interval="${PHYSICS_CLIENT_POLL_INTERVAL_S:-2}"
  local deadline remaining call_timeout sleep_time value
  for value in "$connect_timeout" "$request_timeout" "$poll_timeout" "$poll_interval"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
      printf 'Physics client timeout values must be positive integers: %s\n' "$value" >&2
      return 2
    fi
  done
  deadline=$((SECONDS + poll_timeout))
  while true; do
    remaining=$((deadline - SECONDS))
    if ((remaining <= 0)); then
      printf 'Timed out waiting for %s session %s to reach a terminal status\n' \
        "$family" "$session_id" >&2
      return 1
    fi
    call_timeout="$request_timeout"
    if ((call_timeout > remaining)); then
      call_timeout="$remaining"
    fi
    if ! status_response=$(curl --connect-timeout "$connect_timeout" \
      --max-time "$call_timeout" -fsS \
      "$BASE_URL/$family/$session_id/status"); then
      printf 'Status request failed for %s session %s\n' \
        "$family" "$session_id" >&2
      return 1
    fi
    if ! status=$(jq -er '.status' <<<"$status_response"); then
      printf 'Malformed status response for %s session %s\n' \
        "$family" "$session_id" >&2
      return 1
    fi
    jq . <<<"$status_response" >&2
    case "$status" in
      completed|failed|cancelled)
        printf '%s\n' "$status"
        return 0
        ;;
      pending|running|cancelling)
        remaining=$((deadline - SECONDS))
        sleep_time="$poll_interval"
        if ((sleep_time > remaining)); then
          sleep_time="$remaining"
        fi
        if ((sleep_time > 0)); then
          sleep "$sleep_time"
        fi
        ;;
      *)
        printf 'Unexpected %s status: %s\n' "$family" "$status" >&2
        return 1
        ;;
    esac
  done
}
