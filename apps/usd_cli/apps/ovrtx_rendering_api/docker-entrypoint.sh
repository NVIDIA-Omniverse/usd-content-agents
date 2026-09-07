#!/usr/bin/env bash
# Start a virtual X display (OVRTX/Vulkan wants a DISPLAY) then launch the API.
set -euo pipefail

: "${PORT:=8000}"
: "${DISPLAY:=:0}"
export DISPLAY

# Xvfb provides an offscreen display for headless GPU rendering.
xvfb_pid=""
service_pid=""

stop_children() {
  [ -z "$service_pid" ] || kill -TERM "$service_pid" 2>/dev/null || true
  [ -z "$xvfb_pid" ] || kill -TERM "$xvfb_pid" 2>/dev/null || true
}

wait_for_children() {
  service_status=0
  if [ -n "$service_pid" ]; then
    wait "$service_pid" || service_status="$?"
    service_pid=""
  fi
  if [ -n "$xvfb_pid" ]; then
    kill -TERM "$xvfb_pid" 2>/dev/null || true
    wait "$xvfb_pid" || true
    xvfb_pid=""
  fi
  return "$service_status"
}

handle_signal() {
  stop_children
  wait_for_children || true
  exit 143
}

trap handle_signal INT TERM

if command -v Xvfb >/dev/null 2>&1; then
  Xvfb "${DISPLAY}" -screen 0 1280x1024x24 >/tmp/xvfb.log 2>&1 &
  xvfb_pid="$!"
  # Give Xvfb a moment to come up before the renderer probes the display.
  for _ in $(seq 1 50); do
    [ -e "/tmp/.X11-unix/X${DISPLAY#:}" ] && break
    sleep 0.1
  done
fi

# Do not exec uvicorn: OVRTX's daemon verifies that its expected parent is not
# PID 1. Keeping this entrypoint as the parent also lets it forward shutdown
# signals and reap both the service and Xvfb.
uvicorn service.main:app --host 0.0.0.0 --port "${PORT}" &
service_pid="$!"
wait_for_children
