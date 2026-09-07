#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

import { preToolUsePythonPolicy } from "./codex_sdk_bridge.mjs";

const REASONING_ADAPTER_HOSTS = new Set(["127.0.0.1", "[::1]"]);
const DEFAULT_TOOLS = [
  "Read",
  "Glob",
  "Grep",
  "LS",
  "TodoWrite",
  "Skill",
];
const SANDBOXED_TOOLS = [...DEFAULT_TOOLS, "Bash"];
const SANDBOXED_PERMISSION_MODES = new Set([
  "acceptEdits",
  "bypassPermissions",
]);
const DEFAULT_MAX_REFERENCE_IMAGE_BYTES = 50 * 1024 * 1024;
const SUPPORTED_CLAUDE_CONFIG_KEYS = new Set([
  "env",
  "maxBudgetUsd",
  "settings",
]);
const SECURITY_CRITICAL_CLAUDE_CONFIG_KEYS = [
  "additionalDirectories",
  "allowedTools",
  "allowDangerouslySkipPermissions",
  "cwd",
  "permissionMode",
  "persistSession",
  "sandbox",
  "settingSources",
  "systemPrompt",
  "tools",
];
const DANGEROUS_CLAUDE_ENV_KEYS = new Set([
  "ALL_PROXY",
  "CONTENT_WORKFLOW_CONTROLLED_ARTIFACT_ROOT",
  "HTTP_PROXY",
  "HTTPS_PROXY",
  "LD_PRELOAD",
  "LD_LIBRARY_PATH",
  "NODE_OPTIONS",
  "NO_PROXY",
  "PATH",
  "PYTHONHOME",
  "PYTHONINSPECT",
  "PYTHONNOUSERSITE",
  "PYTHONPATH",
  "PYTHONSAFEPATH",
  "PYTHONSTARTUP",
  "PYTHONUSERBASE",
  "PYTHONWARNINGS",
  "3DSC_NO_DAEMON",
  "OV_NO_DAEMON",
  "USD_CLI_NO_DAEMON",
  "USD_CLI_AGENT",
  "USD_CLI_REMOTE_RENDER_STAGING_ROOT",
  "WARP_CACHE_PATH",
  "XDG_CACHE_HOME",
  "all_proxy",
  "http_proxy",
  "https_proxy",
  "no_proxy",
]);
const USD_CLI_CREDENTIAL_ENV_KEYS = new Set([
  "OVRTX_API_KEY",
  "3DSC_RENDER_REMOTE_API_KEY",
  "OV_RENDER_REMOTE_API_KEY",
  "USD_CLI_RENDER_REMOTE_API_KEY",
  "3DSC_RENDER_BACKEND_API_KEYS_JSON",
  "OV_RENDER_BACKEND_API_KEYS_JSON",
  "USD_CLI_RENDER_BACKEND_API_KEYS_JSON",
  "USD_CLI_TOKEN",
  "USD_CLI_SERVER_TOKEN",
  "OV_TOKEN",
  "OV_SERVER_TOKEN",
  "3DSC_TOKEN",
  "3DSC_SERVER_TOKEN",
  "CONTENT_WORKFLOW_PARENT_USD_CLI_TOKEN",
]);
const BASE_SYSTEM_PROMPT_APPEND =
  "You are running as a non-interactive child agent inside content-workflow-cli. " +
  "Follow the user prompt artifact contract exactly. " +
  "You are a single continuous turn with no later turn to deliver asynchronous " +
  "notifications: tools like Monitor are not in your allowed toolset, and " +
  "backgrounding a Bash command (run_in_background) will not report its result " +
  "back to you either. To wait on a long-running command (for example a batch " +
  "job), run it as one blocking Bash call, such as a shell loop that polls and " +
  "sleeps until the work is done (e.g. `until <condition>; do sleep N; done`), " +
  "or simply run it in the foreground and wait for it to exit. " +
  "Bash commands run in a mandatory OS sandbox: use Bash for selected " +
  "scene-backend requests and for creating artifacts inside the run directory. " +
  "The sandbox blocks writes outside that directory; only the configured " +
  "scene-backend hosts are pre-authorized for Bash network access. Use sandboxed " +
  "Bash to read input paths outside the run directory; those paths are not " +
  "added as writable Claude workspaces.";
const MATERIAL_USD_CLI_SYSTEM_PROMPT_APPEND =
  "Use the workflow-selected usd-cli scene backend through the package-owned usd-cli-tel executable, require OVRTX for renders, and do not modify source USD files.";

async function main() {
  const requestPath = process.argv[2];
  if (!requestPath) {
    throw new Error("Usage: claude_bridge.mjs <request.json>");
  }

  const request = readJsonRequest(requestPath);
  const { query } = await loadClaudeAgentSdk();
  const messages = [];
  let finalResponse = "";
  let resultMessage = null;
  const prompt = await buildPrompt(request);
  validateClaudePrompt(prompt);
  // Do not create the adopted child-final artifact until the structured result
  // is known to be present.  Evidence remains available for failed turns.
  const artifacts = prepareRunArtifacts(request, false);
  let observableArtifact = null;

  try {
    observableArtifact = request.observable_events_path
      ? prepareRunArtifact(request, request.observable_events_path)
      : null;
    for await (const message of query({
      prompt,
      options: buildOptions(request),
    })) {
      messages.push(toJsonable(message));
      writeProgress(message);
      const insight = observableInsightFromMessage(message);
      if (insight && observableArtifact) {
        appendPreparedRunArtifact(
          observableArtifact,
          JSON.stringify(sanitizeObservableRecord(insight)) + "\n",
        );
      }
      if (message?.type === "result") {
        resultMessage = message;
        finalResponse = finalResponseFromResult(message, request.output_schema);
      }
    }

    const missingStructuredOutput = !finalResponse && request.output_schema;
    if (!finalResponse && !missingStructuredOutput) {
      finalResponse = collectAssistantText(messages);
    }

    writePreparedRunArtifact(
      artifacts[0],
      JSON.stringify(toJsonable(messages), null, 2),
    );
    if (request.result_path) {
      writePreparedRunArtifact(
        artifacts[1],
        JSON.stringify(toJsonable(resultMessage ?? {}), null, 2),
      );
    }

    if (missingStructuredOutput) {
      throw new Error(
        "Claude structured-output turn completed without structured_output",
      );
    }

    const finalArtifact = prepareRunArtifact(request, request.child_final_path);
    try {
      writePreparedRunArtifact(finalArtifact, finalResponse);
    } finally {
      fs.closeSync(finalArtifact.fd);
    }

    if (finalResponse) {
      process.stdout.write(finalResponse);
      if (!finalResponse.endsWith("\n")) {
        process.stdout.write("\n");
      }
    }
  } finally {
    for (const artifact of artifacts) {
      fs.closeSync(artifact.fd);
    }
    if (observableArtifact) {
      fs.closeSync(observableArtifact.fd);
    }
  }
}

export function observableInsightFromMessage(message) {
  if (message?.type !== "assistant") {
    return null;
  }
  const blocks = Array.isArray(message.message?.content)
    ? message.message.content
    : [];
  const publicText = blocks
    .filter((block) => block?.type === "text")
    .map((block) => String(block.text ?? "").trim())
    .filter(Boolean)
    .join("\n");
  const toolNames = blocks
    .filter((block) => block?.type === "tool_use")
    .map((block) => String(block.name ?? "tool"));
  if (!publicText && toolNames.length === 0) {
    return null;
  }
  return {
    schema_version: "content-agents.observable-insight.v1",
    id: String(message.uuid ?? message.message?.id ?? `assistant-${Date.now()}`),
    time: new Date().toISOString(),
    phase: "reasoning",
    source: "claude_sdk",
    kind: publicText ? "commentary" : "action",
    title: publicText ? "Agent update" : "Agent tool call",
    summary: boundedObservableText(publicText || toolNames.join(", ")),
    status: "success",
  };
}

function boundedObservableText(value, limit = 1200) {
  const redacted = String(value ?? "")
    .replace(
      /\b(?:Basic|Bearer|Token)\s+(?:"[^"]*"|'[^']*'|`[^`]*`|[^\s,;&]+)/gi,
      "credential [redacted]",
    )
    .replace(
      /\b([a-z0-9_-]*(?:api[_-]?key|authorization|cookie|credential|password|secret|signature|token)[a-z0-9_-]*)\b\s*([=:])\s*(?:"[^"]*"|'[^']*'|`[^`]*`|[^\s,;&]+)/gi,
      "$1$2[redacted]",
    )
    .replace(
      /(--(?:api[-_]?key|authorization|cookie|credential|password|secret|signature|token))(?:\s+|=)(?:"[^"]*"|'[^']*'|[^\s,;&]+)/gi,
      "$1 [redacted]",
    )
    .replace(
      /\b([A-Z][A-Z0-9_]*(?:API_KEY|ACCESS_KEY|SECRET_KEY|SECRET|SIGNATURE|TOKEN|PASSWORD|COOKIE|CREDENTIAL|AUTHORIZATION))\s*=\s*(?:"[^"]*"|'[^']*'|[^\s,;&]+)/g,
      "$1=[redacted]",
    )
    .trim();
  return redacted.length <= limit ? redacted : `${redacted.slice(0, limit)}…`;
}

export function sanitizeObservableRecord(value, depth = 0) {
  if (depth >= 32) {
    return "[truncated]";
  }
  if (Array.isArray(value)) {
    return value.map((item) => sanitizeObservableRecord(item, depth + 1));
  }
  if (value && typeof value === "object") {
    const sanitized = {};
    for (const [key, item] of Object.entries(value)) {
      const canonicalKey = key.toLowerCase().replace(/[^a-z0-9]/g, "");
      sanitized[key] = /apikey|accesskey|authorization|cookie|credential|password|secret|signature|token/.test(
        canonicalKey,
      )
        ? "[redacted]"
        : sanitizeObservableRecord(item, depth + 1);
    }
    return sanitized;
  }
  if (typeof value === "string") {
    return boundedObservableText(value);
  }
  return value;
}

export function writeRunArtifact(request, filePath, content) {
  const artifact = prepareRunArtifact(request, filePath);
  try {
    writePreparedRunArtifact(artifact, content);
  } finally {
    fs.closeSync(artifact.fd);
  }
}

export function prepareRunArtifacts(request, includeChildFinal = true) {
  const paths = [
    ...(includeChildFinal ? [request.child_final_path] : []),
    request.items_path,
    ...(request.result_path ? [request.result_path] : []),
  ];
  const artifacts = [];
  try {
    for (const filePath of paths) {
      artifacts.push(prepareRunArtifact(request, filePath));
    }
    return artifacts;
  } catch (error) {
    for (const artifact of artifacts) {
      fs.closeSync(artifact.fd);
    }
    throw error;
  }
}

function containedWithin(rootPath, resolvedPath) {
  const lexicalRoot = path.resolve(rootPath);
  const realRoot = fs.realpathSync(lexicalRoot);
  if (lexicalRoot !== realRoot) {
    throw new Error(`Bridge artifact root must not be a symlink: ${lexicalRoot}`);
  }
  const relativePath = path.relative(lexicalRoot, resolvedPath);
  return !(
    !relativePath ||
    relativePath === ".." ||
    relativePath.startsWith(`..${path.sep}`) ||
    path.isAbsolute(relativePath)
  );
}

export function prepareRunArtifact(request, filePath) {
  const resolvedPath = path.resolve(filePath);
  // The runner stages host-bridge outputs outside the child-writable run
  // directory on purpose, so confinement is checked against the run directory
  // or the single staging root the runner declared -- never an arbitrary path.
  const roots = [request.run_dir];
  if (request.bridge_staging_root) {
    roots.push(request.bridge_staging_root);
  }
  if (!roots.some((root) => containedWithin(root, resolvedPath))) {
    throw new Error(
      `Bridge artifact must stay inside run_dir or the declared staging root: ${filePath}`,
    );
  }
  const parentPath = path.dirname(resolvedPath);
  const realParentPath = fs.realpathSync(parentPath);
  if (realParentPath !== parentPath) {
    throw new Error(`Bridge artifact parent must not contain symlinks: ${parentPath}`);
  }

  try {
    const existing = fs.lstatSync(resolvedPath);
    if (existing.isDirectory()) {
      throw new Error(`Bridge artifact path is a directory: ${resolvedPath}`);
    }
    fs.unlinkSync(resolvedPath);
  } catch (error) {
    if (error?.code !== "ENOENT") {
      throw error;
    }
  }

  const flags =
    fs.constants.O_WRONLY |
    fs.constants.O_CREAT |
    fs.constants.O_EXCL |
    (fs.constants.O_NOFOLLOW ?? 0);
  const fd = fs.openSync(resolvedPath, flags, 0o600);
  const metadata = fs.fstatSync(fd);
  return {
    fd,
    path: resolvedPath,
    device: metadata.dev,
    inode: metadata.ino,
  };
}

function assertPreparedRunArtifact(artifact) {
  let current;
  try {
    current = fs.lstatSync(artifact.path);
  } catch (error) {
    throw new Error(`Bridge artifact path changed during child turn: ${artifact.path}`, {
      cause: error,
    });
  }
  if (
    !current.isFile() ||
    current.isSymbolicLink() ||
    current.dev !== artifact.device ||
    current.ino !== artifact.inode ||
    current.nlink !== 1
  ) {
    throw new Error(`Bridge artifact path changed during child turn: ${artifact.path}`);
  }
}

export function writePreparedRunArtifact(artifact, content) {
  assertPreparedRunArtifact(artifact);
  fs.ftruncateSync(artifact.fd, 0);
  fs.writeFileSync(artifact.fd, content, "utf8");
  fs.fchmodSync(artifact.fd, 0o600);
  fs.fsyncSync(artifact.fd);
  assertPreparedRunArtifact(artifact);
}

export function appendPreparedRunArtifact(artifact, content) {
  assertPreparedRunArtifact(artifact);
  fs.writeSync(artifact.fd, content, null, "utf8");
  fs.fchmodSync(artifact.fd, 0o600);
  fs.fsyncSync(artifact.fd);
  assertPreparedRunArtifact(artifact);
}

export function readJsonRequest(requestPath) {
  try {
    return JSON.parse(fs.readFileSync(requestPath, "utf8"));
  } catch (error) {
    throw new Error(
      `Invalid Claude bridge request file at ${requestPath}: ${error.message}`,
    );
  }
}

async function loadClaudeAgentSdk() {
  try {
    return await import("@anthropic-ai/claude-agent-sdk");
  } catch (error) {
    // `npm ci` installs the exact lockfile version the sandbox mask allowlist
    // in runner.py was validated against. `npm install` can resolve a
    // different SDK build whose mask surface has drifted.
    process.stderr.write(
      "Unable to import @anthropic-ai/claude-agent-sdk. From the repository root, run `npm ci --prefix agentic/packages/content_workflow_cli`.\n",
    );
    throw error;
  }
}

export function dropForbiddenEnvironmentNames(
  environment,
  forbiddenNames,
  platform = process.platform,
) {
  for (const environmentName of forbiddenNames ?? []) {
    if (typeof environmentName !== "string") continue;
    if (platform === "win32") {
      const canonicalName = environmentName.toUpperCase();
      for (const key of Object.keys(environment)) {
        if (key.toUpperCase() === canonicalName) delete environment[key];
      }
    } else {
      delete environment[environmentName];
    }
  }
}

export function buildOptions(request) {
  const claudeConfig = { ...(request.claude_config ?? {}) };
  dropSecurityCriticalClaudeConfigKeys(claudeConfig);
  dropUnsupportedClaudeConfigKeys(claudeConfig);
  sanitizeClaudeConfigSettings(claudeConfig);
  const permissionMode = request.claude_permission_mode ?? "default";
  const rawConfigEnv =
    claudeConfig.env &&
    typeof claudeConfig.env === "object" &&
    !Array.isArray(claudeConfig.env)
      ? claudeConfig.env
      : {};
  const configEnv = filterClaudeConfigEnv(rawConfigEnv);
  const childEnv = {
    ...process.env,
    CLAUDE_AGENT_SDK_CLIENT_APP: "nvidia-content-workflow-cli/0.1.0",
    ...configEnv,
  };
  if (process.platform === "win32" && typeof process.env.PATH === "string") {
    // Windows environment names are case-insensitive, but object spread keeps
    // the host's display spelling (commonly `Path`). Expose one canonical key
    // without allowing claude_config.env to replace the trusted host value.
    for (const key of Object.keys(childEnv)) {
      if (key !== "PATH" && key.toUpperCase() === "PATH") delete childEnv[key];
    }
    childEnv.PATH = process.env.PATH;
  }
  dropForbiddenEnvironmentNames(
    childEnv,
    request.child_launch?.credential_policy?.forbidden_environment_names,
  );
  delete claudeConfig.env;
  // repo_root is always the run directory (_agent_working_directory confines
  // every child runner there), which is also where _stage_agent_skills copies
  // the trusted skills the child discovers.
  const childCwd = request.repo_root;
  const options = {
    cwd: childCwd,
    env: childEnv,
    ...claudeConfig,
    allowedTools: request.tools_disabled ? [] : DEFAULT_TOOLS,
    permissionMode,
    allowDangerouslySkipPermissions: permissionMode === "bypassPermissions",
    persistSession: false,
    sandbox: buildSandboxSettings(request),
    hooks: {
      PreToolUse: [
        {
          matcher: "Bash",
          hooks: [
            async (input) =>
              preToolUsePythonPolicy(
                input,
                request.trusted_repo_root ?? null,
                null,
              ),
          ],
        },
      ],
    },
    // "project" enables discovery of the skills staged under
    // <run_dir>/.claude/skills. The project root is the child-writable run
    // directory, so child-authored project instructions, hooks, or permission
    // allowlists could otherwise survive into the next turn.
    // _sanitize_child_project_surfaces purges all child-owned provider project
    // trees before each launch, and _stage_agent_skills then restores only the
    // trusted skill catalog required by that turn.
    settingSources: ["project"],
    systemPrompt: {
      type: "preset",
      preset: "claude_code",
      append: buildSystemPromptAppend(request),
    },
  };
  // acceptEdits and bypassPermissions auto-approve direct file mutations.
  // Restrict the SDK's actual tool surface in both modes to read-only tools
  // plus Bash. The mandatory OS sandbox confines Bash writes and network.
  if (request.tools_disabled) {
    options.tools = [];
  } else if (SANDBOXED_PERMISSION_MODES.has(permissionMode)) {
    options.tools = SANDBOXED_TOOLS;
  } else {
    delete options.tools;
  }
  const additionalDirectories = sanitizeAdditionalDirectories(
    request.additional_directories,
  );
  if (additionalDirectories.length > 0) {
    // The run directory can live outside the agent working directory; grant
    // it explicitly so the child can write the required run artifacts.
    options.additionalDirectories = additionalDirectories;
  }
  if (request.model) {
    options.model = request.model;
  }
  const effort = mapEffort(request.model_reasoning_effort);
  if (effort) {
    options.effort = effort;
  }
  if (request.claude_max_turns) {
    options.maxTurns = request.claude_max_turns;
  }
  if (request.output_schema) {
    options.outputFormat = {
      type: "json_schema",
      schema: request.output_schema,
    };
  }
  return options;
}

export function sanitizeAdditionalDirectories(value) {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter(
    (dir) => typeof dir === "string" && dir.length > 0 && path.isAbsolute(dir),
  );
}

function buildSandboxSettings(request) {
  const launchNetworkPolicy = request.child_launch?.network_policy;
  if (launchNetworkPolicy !== undefined) {
    const allowedDomains = Array.isArray(launchNetworkPolicy.allowed_hosts)
      ? launchNetworkPolicy.allowed_hosts.filter(
          (host) => typeof host === "string" && host.length > 0,
        )
      : [];
    if (launchNetworkPolicy.mode === "reasoning_transport_only") {
      if (
        launchNetworkPolicy.tool_network_access === true ||
        allowedDomains.some((host) => !REASONING_ADAPTER_HOSTS.has(host))
      ) {
        throw new Error(
          "Reasoning-only child launch network policy can allow only loopback adapters",
        );
      }
    } else if (
      launchNetworkPolicy.tool_network_access !== true &&
      allowedDomains.length > 0
    ) {
      throw new Error(
        "Child launch network policy cannot allow domains when tool network is disabled",
      );
    }
    return {
      enabled: true,
      failIfUnavailable: true,
      autoAllowBashIfSandboxed: true,
      allowUnsandboxedCommands: false,
      network: { allowedDomains },
    };
  }
  const allowedDomains = [];
  if (
    request.scene_backend === "usd-cli" &&
    !allowedDomains.includes("127.0.0.1")
  ) {
    // The package-owned CLI talks to the wrapper-started, identity-checked
    // project daemon on this fixed loopback host.
    allowedDomains.push("127.0.0.1");
  }
  // The tuning broker listens on loopback at an ephemeral port. With the
  // usd-cli daemon host the endpoints coincide, but an additional remote host
  // would otherwise leave the broker outside the allowlist and every sweep
  // would exit 6 ("broker unreachable").
  for (const host of request.extra_allowed_hosts ?? []) {
    if (typeof host === "string" && host && !allowedDomains.includes(host)) {
      allowedDomains.push(host);
    }
  }
  return {
    enabled: true,
    failIfUnavailable: true,
    autoAllowBashIfSandboxed: true,
    allowUnsandboxedCommands: false,
    network: { allowedDomains },
  };
}

export function buildSystemPromptAppend(request) {
  if (request.workflow === "materials.assign") {
    return `${BASE_SYSTEM_PROMPT_APPEND} ${MATERIAL_USD_CLI_SYSTEM_PROMPT_APPEND}`;
  }
  return BASE_SYSTEM_PROMPT_APPEND;
}

function filterClaudeConfigEnv(configEnv) {
  const filtered = {};
  const droppedKeys = [];
  for (const [key, value] of Object.entries(configEnv)) {
    const canonicalKey = key.toUpperCase();
    if (
      DANGEROUS_CLAUDE_ENV_KEYS.has(canonicalKey) ||
      USD_CLI_CREDENTIAL_ENV_KEYS.has(canonicalKey) ||
      canonicalKey === "TRACEPARENT" ||
      canonicalKey === "TRACESTATE" ||
      canonicalKey === "USD_CLI_EXTERNAL_LIFECYCLE_BOOTSTRAP" ||
      canonicalKey === "USD_CLI_LIFECYCLE_EXTERNALLY_OWNED" ||
      canonicalKey.startsWith("CONTENT_WORKFLOW_PARENT_USD_CLI_") ||
      canonicalKey.startsWith("CONTENT_WORKFLOW_USD_CLI_") ||
      canonicalKey.startsWith("PYTHON") ||
      canonicalKey.startsWith("USD_CLI_SERVER_") ||
      canonicalKey.startsWith("USD_CLI_TEL_")
    ) {
      droppedKeys.push(key);
      continue;
    }
    filtered[key] = value;
  }
  if (droppedKeys.length > 0) {
    process.stderr.write(
      `Ignoring dangerous Claude config env key(s): ${droppedKeys.join(", ")}.\n`,
    );
  }
  return filtered;
}

function dropSecurityCriticalClaudeConfigKeys(claudeConfig) {
  const droppedKeys = SECURITY_CRITICAL_CLAUDE_CONFIG_KEYS.filter(
    (key) => Object.hasOwn(claudeConfig, key),
  );
  if (droppedKeys.length === 0) {
    return;
  }
  for (const key of droppedKeys) {
    delete claudeConfig[key];
  }
  process.stderr.write(
    `Ignoring security-critical Claude config key(s): ${droppedKeys.join(", ")}.\n`,
  );
}

function dropUnsupportedClaudeConfigKeys(claudeConfig) {
  const droppedKeys = Object.keys(claudeConfig).filter(
    (key) => !SUPPORTED_CLAUDE_CONFIG_KEYS.has(key),
  );
  if (droppedKeys.length === 0) {
    return;
  }
  for (const key of droppedKeys) {
    delete claudeConfig[key];
  }
  process.stderr.write(
    `Ignoring unsupported Claude config key(s): ${droppedKeys.join(", ")}. ` +
      "Only env, maxBudgetUsd, and settings are supported for content-workflow-cli.\n",
  );
}

function sanitizeClaudeConfigSettings(claudeConfig) {
  if (!Object.hasOwn(claudeConfig, "settings")) {
    return;
  }
  const settings = claudeConfig.settings;
  if (!settings || typeof settings !== "object" || Array.isArray(settings)) {
    delete claudeConfig.settings;
    process.stderr.write(
      "Ignoring Claude config settings because content-workflow-cli only accepts object settings.\n",
    );
    return;
  }

  const sanitizedSettings = {};
  const droppedSettingKeys = Object.keys(settings).filter(
    (key) => key !== "permissions",
  );
  const permissions = settings.permissions;
  if (
    permissions &&
    typeof permissions === "object" &&
    !Array.isArray(permissions)
  ) {
    const safePermissionKeys = new Set([
      "ask",
      "deny",
      "disableBypassPermissionsMode",
    ]);
    const sanitizedPermissions = {};
    const droppedKeys = [];
    for (const [key, value] of Object.entries(permissions)) {
      if (safePermissionKeys.has(key)) {
        sanitizedPermissions[key] = value;
      } else {
        droppedKeys.push(key);
      }
    }
    if (Object.keys(sanitizedPermissions).length > 0) {
      sanitizedSettings.permissions = sanitizedPermissions;
    }
    if (droppedKeys.length > 0) {
      process.stderr.write(
        "Ignoring Claude config settings.permissions key(s) that can expand " +
          `tool access: ${droppedKeys.join(", ")}.\n`,
      );
    }
  }
  if (droppedSettingKeys.length > 0) {
    process.stderr.write(
      "Ignoring Claude config settings key(s) that are not permission " +
        `tightening controls: ${droppedSettingKeys.join(", ")}.\n`,
    );
  }
  if (Object.keys(sanitizedSettings).length === 0) {
    delete claudeConfig.settings;
  } else {
    claudeConfig.settings = sanitizedSettings;
  }
}

function mapEffort(value) {
  if (!value) {
    return undefined;
  }
  if (value === "minimal") {
    return "low";
  }
  return value;
}

export async function buildPrompt(request) {
  const content = [{ type: "text", text: request.prompt }];
  for (const imagePath of request.reference_images ?? []) {
    content.push({ type: "text", text: `Reference image: ${path.basename(imagePath)}` });
    const image = await readImageBlock(imagePath);
    if (image) {
      content.push(image);
    }
  }
  for (const filePath of request.reference_files ?? []) {
    await validateReadableFile(filePath, "reference file");
    content.push({
      type: "text",
      text: `Reference file: ${path.basename(filePath)} (${filePath})`,
    });
  }
  for (const promptImage of request.prompt_image_inputs ?? []) {
    const imagePath = promptImage?.path;
    if (!imagePath) {
      continue;
    }
    const label = promptImage?.label ?? "Prompt image";
    content.push({ type: "text", text: `${label}: ${path.basename(imagePath)}` });
    const image = await readImageBlock(imagePath);
    if (image) {
      content.push(image);
    }
  }
  if (content.length === 1) {
    return request.prompt;
  }
  // The Claude Agent SDK query API accepts AsyncIterable<SDKUserMessage> for
  // multimodal user messages. Use an explicit iterable when images are attached.
  return userMessageStream(content);
}

function validateClaudePrompt(prompt) {
  if (
    typeof prompt !== "string" &&
    typeof prompt?.[Symbol.asyncIterator] !== "function"
  ) {
    throw new Error("Claude prompt must be a string or async iterable message stream");
  }
}

async function* userMessageStream(content) {
  yield {
    type: "user",
    message: {
      role: "user",
      content,
    },
    parent_tool_use_id: null,
  };
}

async function readImageBlock(imagePath) {
  let stat;
  try {
    await fs.promises.access(imagePath, fs.constants.R_OK);
    stat = await fs.promises.stat(imagePath);
  } catch (error) {
    throw new Error(`Unable to read reference image: ${imagePath}`, {
      cause: error,
    });
  }
  if (!stat.isFile()) {
    throw new Error(`Reference image is not a file: ${imagePath}`);
  }
  const mediaType = mediaTypeForPath(imagePath);
  if (!mediaType) {
    throw new Error(`Unsupported reference image type: ${imagePath}`);
  }
  const maxBytes = maxReferenceImageBytes();
  if (stat.size > maxBytes) {
    throw new Error(`Reference image exceeds ${maxBytes} bytes: ${imagePath}`);
  }
  return {
    type: "image",
    source: {
      type: "base64",
      media_type: mediaType,
      data: (await fs.promises.readFile(imagePath)).toString("base64"),
    },
  };
}

async function validateReadableFile(filePath, label) {
  let stat;
  try {
    await fs.promises.access(filePath, fs.constants.R_OK);
    stat = await fs.promises.stat(filePath);
  } catch (error) {
    throw new Error(`Unable to read ${label}: ${filePath}`, { cause: error });
  }
  if (!stat.isFile()) {
    throw new Error(`${label} is not a file: ${filePath}`);
  }
}

function maxReferenceImageBytes() {
  const configured = Number.parseInt(
    process.env.CONTENT_AGENTS_MAX_REFERENCE_IMAGE_BYTES ?? "",
    10,
  );
  if (Number.isFinite(configured) && configured > 0) {
    return configured;
  }
  return DEFAULT_MAX_REFERENCE_IMAGE_BYTES;
}

function mediaTypeForPath(imagePath) {
  const extension = path.extname(imagePath).toLowerCase();
  if (extension === ".png") {
    return "image/png";
  }
  if (extension === ".jpg" || extension === ".jpeg") {
    return "image/jpeg";
  }
  if (extension === ".webp") {
    return "image/webp";
  }
  if (extension === ".gif") {
    return "image/gif";
  }
  return null;
}

function writeProgress(message) {
  if (message?.type === "assistant") {
    for (const block of message.message?.content ?? []) {
      if (block?.type === "text" && block.text) {
        process.stdout.write(block.text);
        if (!block.text.endsWith("\n")) {
          process.stdout.write("\n");
        }
      } else if (block?.type === "tool_use") {
        process.stdout.write(`Tool: ${block.name}\n`);
      }
    }
  } else if (message?.type === "result") {
    process.stdout.write(`Claude result: ${message.subtype}\n`);
  }
}

/** Return the child-final artifact content for one Claude result event. */
export function finalResponseFromResult(message, outputSchema) {
  if (outputSchema) {
    if (message?.structured_output === undefined) {
      return "";
    }
    return JSON.stringify(message.structured_output);
  }
  return String(message?.result ?? "");
}

function collectAssistantText(messages) {
  const chunks = [];
  for (const message of messages) {
    if (message?.type !== "assistant") {
      continue;
    }
    for (const block of message.message?.content ?? []) {
      if (block?.type === "text" && block.text) {
        chunks.push(block.text);
      }
    }
  }
  return chunks.join("\n");
}

function toJsonable(value, seen = new WeakSet()) {
  if (value === null || value === undefined) {
    return value;
  }
  if (typeof value !== "object") {
    return value;
  }
  if (seen.has(value)) {
    return "[Circular]";
  }
  seen.add(value);
  if (Array.isArray(value)) {
    return value.map((item) => toJsonable(item, seen));
  }
  const output = {};
  for (const [key, item] of Object.entries(value)) {
    if (typeof item !== "function") {
      output[key] = toJsonable(item, seen);
    }
  }
  return output;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    process.stderr.write(`${error.stack ?? error}\n`);
    process.exitCode = 1;
  });
}
