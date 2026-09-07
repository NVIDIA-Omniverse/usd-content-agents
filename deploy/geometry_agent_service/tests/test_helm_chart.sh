#!/usr/bin/env bash
set -euo pipefail

chart_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

assert_env_value() {
  local rendered="$1"
  local name="$2"
  local expected="$3"
  grep -A1 -F "name: $name" <<<"$rendered" | grep -Fq "value: \"$expected\""
}

helm lint "$chart_dir" \
  --set image.repository=example/geometry-agent-service \
  --set apiKey=development-only-key

rendered="$(helm template geometry "$chart_dir" \
  --set image.repository=example/geometry-agent-service \
  --set existingApiKeySecretName=geometry-api-key)"

grep -q 'GEOMETRY_AGENT_SERVICE_API_KEY' <<<"$rendered"
grep -q 'GEOMETRY_AGENT_SERVICE_RENDER_BACKEND' <<<"$rendered"
assert_env_value "$rendered" GEOMETRY_AGENT_SERVICE_DEFAULT_RUNTIME_ENGINE none
assert_env_value "$rendered" HOME /var/cache/geometry-agent-service/home
assert_env_value "$rendered" XDG_CACHE_HOME /var/cache/geometry-agent-service/xdg
assert_env_value "$rendered" UV_CACHE_DIR /var/cache/geometry-agent-service/uv
assert_env_value "$rendered" WU_OVRTX_VENV_DIR /var/cache/geometry-agent-service/ovrtx-venv
grep -A1 -F 'name: ovrtx-cache' <<<"$rendered" \
  | grep -Fq 'mountPath: /var/cache/geometry-agent-service'
grep -A1 -F 'name: ovrtx-cache' <<<"$rendered" | grep -Fq 'emptyDir: {}'
grep -q 'readOnlyRootFilesystem: true' <<<"$rendered"
grep -q 'path: /api/health' <<<"$rendered"

remote_rendered="$(helm template geometry "$chart_dir" \
  --set image.repository=example/geometry-agent-service \
  --set existingApiKeySecretName=geometry-api-key \
  --set render.backend=remote \
  --set render.remoteBaseUrl=https://render.example/v1 \
  --set render.existingApiKeySecretName=render-api-key)"

grep -q 'GEOMETRY_AGENT_SERVICE_RENDER_REMOTE_BASE_URL' <<<"$remote_rendered"
grep -q 'https://render.example/v1' <<<"$remote_rendered"
grep -q 'GEOMETRY_AGENT_SERVICE_RENDER_REMOTE_API_KEY' <<<"$remote_rendered"
if grep -q 'WU_OVRTX_VENV_DIR\|name: ovrtx-cache' <<<"$remote_rendered"; then
  echo "remote rendering unexpectedly received a local OVRTX cache" >&2
  exit 1
fi

unauthenticated_sidecar_rendered="$(helm template geometry "$chart_dir" \
  --set image.repository=example/geometry-agent-service \
  --set existingApiKeySecretName=geometry-api-key \
  --set render.backend=remote \
  --set render.remoteBaseUrl=http://trusted-render-sidecar:8011 \
  --set render.allowUnauthenticatedIdentity=true)"

assert_env_value "$unauthenticated_sidecar_rendered" \
  GEOMETRY_AGENT_SERVICE_RENDER_REMOTE_ALLOW_UNAUTHENTICATED_IDENTITY true
if grep -q 'GEOMETRY_AGENT_SERVICE_RENDER_REMOTE_API_KEY' <<<"$unauthenticated_sidecar_rendered"; then
  echo "unauthenticated remote sidecar unexpectedly received an API key" >&2
  exit 1
fi

if helm template geometry "$chart_dir" \
  --set image.repository=example/geometry-agent-service \
  --set existingApiKeySecretName=geometry-api-key \
  --set render.backend=remote >/dev/null 2>&1; then
  echo "chart unexpectedly accepted remote rendering without an endpoint" >&2
  exit 1
fi

if helm template geometry "$chart_dir" \
  --set image.repository=example/geometry-agent-service \
  --set existingApiKeySecretName=geometry-api-key \
  --set render.allowUnauthenticatedIdentity=true >/dev/null 2>&1; then
  echo "chart unexpectedly accepted unauthenticated remote identity without remote rendering" >&2
  exit 1
fi

if helm template geometry "$chart_dir" \
  --set image.repository=example/geometry-agent-service \
  --set existingApiKeySecretName=geometry-api-key \
  --set render.backend=remote \
  --set render.remoteBaseUrl=https://render.example/v1 >/dev/null 2>&1; then
  echo "chart unexpectedly accepted remote rendering without an API key" >&2
  exit 1
fi

if helm template geometry "$chart_dir" \
  --set image.repository=example/geometry-agent-service \
  --set apiKey=development-only-key \
  --set runtime.defaultEngine=unsupported >/dev/null 2>&1; then
  echo "chart unexpectedly accepted an unsupported runtime engine" >&2
  exit 1
fi

if helm template geometry "$chart_dir" \
  --set image.repository=example/geometry-agent-service \
  --set existingApiKeySecretName=geometry-api-key \
  --set replicaCount=2 >/dev/null 2>&1; then
  echo "chart unexpectedly accepted multiple replicas with an isolated workspace" >&2
  exit 1
fi

if helm template geometry "$chart_dir" \
  --set image.repository=example/geometry-agent-service \
  --set existingApiKeySecretName=geometry-api-key \
  --set replicaCount=2 \
  --set workspace.kind=pvc >/dev/null 2>&1; then
  echo "chart unexpectedly created a single-writer claim for multiple replicas" >&2
  exit 1
fi

shared_rendered="$(helm template geometry "$chart_dir" \
  --set image.repository=example/geometry-agent-service \
  --set existingApiKeySecretName=geometry-api-key \
  --set replicaCount=2 \
  --set workspace.kind=pvc \
  --set workspace.existingClaim=geometry-shared-workspace)"

grep -q 'replicas: 2' <<<"$shared_rendered"
grep -q 'claimName: geometry-shared-workspace' <<<"$shared_rendered"

generic_delegated_rendered="$(helm template geometry "$chart_dir" \
  --set image.repository=example/geometry-agent-service \
  --set existingApiKeySecretName=geometry-api-key \
  --set delegatedAuthoring.enabled=true \
  --set delegatedAuthoring.providerId=private-authoring-worker \
  --set delegatedAuthoring.providerLabel='Private authoring worker' \
  --set delegatedAuthoring.endpointUrl=https://workers.example/private-authoring \
  --set delegatedAuthoring.rightsAssertion=authorized \
  --set delegatedAuthoring.existingTokenSecretName=private-worker-token)"

grep -q 'GEOMETRY_AGENT_SERVICE_DELEGATED_AUTHORING_PROVIDER_ID' \
  <<<"$generic_delegated_rendered"
grep -q 'private-authoring-worker' <<<"$generic_delegated_rendered"
grep -q 'GEOMETRY_AGENT_SERVICE_DELEGATED_AUTHORING_BEARER_TOKEN' \
  <<<"$generic_delegated_rendered"
grep -q 'GEOMETRY_AGENT_SERVICE_DELEGATED_AUTHORING_SUPPORTS_EXPORT' \
  <<<"$generic_delegated_rendered"

build123d_rendered="$(helm template geometry "$chart_dir" \
  --set image.repository=example/geometry-agent-service \
  --set existingApiKeySecretName=geometry-api-key \
  --set build123d.enabled=true \
  --set build123d.endpointUrl=https://workers.example/build123d \
  --set build123d.rightsAssertion=authorized \
  --set build123d.existingTokenSecretName=build123d-worker-token \
  --set 'build123d.supportedFormats={step,stl}' \
  --set build123d.supportsRevision=true \
  --set build123d.supportsExport=true \
  --set build123d.supportsSemanticParameters=true \
  --set build123d.supportsParameterDefinitions=true \
  --set build123d.supportsSemanticParts=true \
  --set build123d.supportsProviderAssertions=true \
  --set build123d.maxFamilyVariants=16)"

assert_env_value "$build123d_rendered" \
  GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTED_FORMATS '[\"step\",\"stl\"]'
assert_env_value "$build123d_rendered" \
  GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTS_TEXT true
assert_env_value "$build123d_rendered" \
  GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTS_IMAGE false
assert_env_value "$build123d_rendered" \
  GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTS_REVISION true
assert_env_value "$build123d_rendered" \
  GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTS_EXPORT true
assert_env_value "$build123d_rendered" \
  GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTS_SEMANTIC_PARAMETERS true
assert_env_value "$build123d_rendered" \
  GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTS_PARAMETER_DEFINITIONS true
assert_env_value "$build123d_rendered" \
  GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTS_SEMANTIC_PARTS true
assert_env_value "$build123d_rendered" \
  GEOMETRY_AGENT_SERVICE_BUILD123D_SUPPORTS_PROVIDER_ASSERTIONS true
assert_env_value "$build123d_rendered" \
  GEOMETRY_AGENT_SERVICE_BUILD123D_MAX_FAMILY_VARIANTS 16

delegated_rendered="$(helm template geometry "$chart_dir" \
  --set image.repository=example/geometry-agent-service \
  --set existingApiKeySecretName=geometry-api-key \
  --set forgecadAuthoring.enabled=true \
  --set forgecadAuthoring.endpointUrl=https://workers.example/forgecad \
  --set forgecadAuthoring.rightsAssertion=authorized \
  --set forgecadAuthoring.automatedUseAuthorized=true \
  --set forgecadAuthoring.existingTokenSecretName=forgecad-worker-token \
  --set forgecadAuthoring.supportsRevision=true \
  --set forgecadAuthoring.supportsExport=true \
  --set forgecadAuthoring.supportsSemanticParameters=true \
  --set forgecadAuthoring.supportsParameterDefinitions=true \
  --set forgecadAuthoring.supportsSemanticParts=true \
  --set forgecadAuthoring.supportsProviderAssertions=true \
  --set forgecadAuthoring.maxFamilyVariants=16 \
  --set forgecadAuthoring.returnsNativeSource=true \
  --set forgecadAuthoring.connectTimeoutSeconds=15 \
  --set forgecadAuthoring.readTimeoutSeconds=900)"

grep -q 'GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_ENDPOINT_URL' <<<"$delegated_rendered"
grep -q 'GEOMETRY_AGENT_SERVICE_FORGECAD_AUTOMATED_USE_AUTHORIZED' <<<"$delegated_rendered"
grep -q 'GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_SUPPORTS_EXPORT' <<<"$delegated_rendered"
assert_env_value "$delegated_rendered" \
  GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_SUPPORTS_SEMANTIC_PARAMETERS true
assert_env_value "$delegated_rendered" \
  GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_SUPPORTS_PARAMETER_DEFINITIONS true
assert_env_value "$delegated_rendered" \
  GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_SUPPORTS_SEMANTIC_PARTS true
assert_env_value "$delegated_rendered" \
  GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_SUPPORTS_PROVIDER_ASSERTIONS true
assert_env_value "$delegated_rendered" \
  GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_MAX_FAMILY_VARIANTS 16
assert_env_value "$delegated_rendered" \
  GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_CONNECT_TIMEOUT_SECONDS 15
assert_env_value "$delegated_rendered" \
  GEOMETRY_AGENT_SERVICE_FORGECAD_AUTHORING_READ_TIMEOUT_SECONDS 900

if helm template geometry "$chart_dir" \
  --set image.repository=example/geometry-agent-service >/dev/null 2>&1; then
  echo "chart unexpectedly accepted a missing API key" >&2
  exit 1
fi

if helm template geometry "$chart_dir" \
  --set image.repository=example/geometry-agent-service \
  --set apiKey=development-only-key \
  --set forgecadAuthoring.enabled=true \
  --set forgecadAuthoring.endpointUrl=https://workers.example/forgecad \
  --set forgecadAuthoring.rightsAssertion=authorized \
  --set forgecadAuthoring.automatedUseAuthorized=true \
  --set forgecadAuthoring.existingTokenSecretName=forgecad-worker-token \
  --set forgecadAuthoring.maxFamilyVariants=3 >/dev/null 2>&1; then
  echo "chart unexpectedly accepted ForgeCAD families without semantic revisions" >&2
  exit 1
fi

if helm template geometry "$chart_dir" \
  --set image.repository=example/geometry-agent-service \
  --set apiKey=development-only-key \
  --set forgecadAuthoring.readTimeoutSeconds=901 >/dev/null 2>&1; then
  echo "chart unexpectedly accepted a ForgeCAD read timeout above 900 seconds" >&2
  exit 1
fi

if helm template geometry "$chart_dir" \
  --set image.repository=example/geometry-agent-service \
  --set apiKey=development-only-key \
  --set delegatedAuthoring.enabled=true \
  --set delegatedAuthoring.endpointUrl=https://workers.example/private-authoring \
  --set delegatedAuthoring.rightsAssertion=authorized \
  --set delegatedAuthoring.existingTokenSecretName=private-worker-token \
  >/dev/null 2>&1; then
  echo "chart unexpectedly accepted delegated authoring without a provider ID" >&2
  exit 1
fi

if helm template geometry "$chart_dir" \
  --set image.repository=example/geometry-agent-service \
  --set apiKey=development-only-key \
  --set delegatedAuthoring.enabled=true \
  --set delegatedAuthoring.providerId=private-authoring-worker \
  --set delegatedAuthoring.endpointUrl=https://workers.example/private-authoring \
  --set delegatedAuthoring.rightsAssertion=authorized \
  >/dev/null 2>&1; then
  echo "chart unexpectedly accepted delegated authoring without a token secret" >&2
  exit 1
fi

if helm template geometry "$chart_dir" \
  --set image.repository=example/geometry-agent-service \
  --set apiKey=development-only-key \
  --set forgecadAuthoring.enabled=true \
  --set forgecadAuthoring.endpointUrl=https://workers.example/forgecad \
  --set forgecadAuthoring.rightsAssertion=authorized \
  --set forgecadAuthoring.existingTokenSecretName=forgecad-worker-token >/dev/null 2>&1; then
  echo "chart unexpectedly accepted ForgeCAD authoring without authorization" >&2
  exit 1
fi

if helm template geometry "$chart_dir" \
  --set image.repository=example/geometry-agent-service \
  --set apiKey=development-only-key \
  --set ingress.enabled=true \
  --set ingress.host=geometry.example.com >/dev/null 2>&1; then
  echo "chart unexpectedly accepted external ingress without TLS" >&2
  exit 1
fi
