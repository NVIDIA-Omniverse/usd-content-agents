#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

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
  "HTTP_PROXY",
  "HTTPS_PROXY",
  "LD_PRELOAD",
  "LD_LIBRARY_PATH",
  "NODE_OPTIONS",
  "NO_PROXY",
  "PATH",
  "all_proxy",
  "http_proxy",
  "https_proxy",
  "no_proxy",
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
  "Bash commands run in a mandatory OS sandbox: use Bash for Content " +
  "Workbench requests and for creating artifacts inside the run directory. " +
  "The sandbox blocks writes outside that directory; only the configured " +
  "Workbench host is pre-authorized for Bash network access. Use sandboxed " +
  "Bash to read input paths outside the run directory; those paths are not " +
  "added as writable Claude workspaces.";
const MATERIAL_SYSTEM_PROMPT_APPEND =
  "Use the Content Workbench API for scene inspection/material edits/renders, and do not modify source USD files.";

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
  const artifacts = prepareRunArtifacts(request);

  try {
    for await (const message of query({
      prompt,
      options: buildOptions(request),
    })) {
      messages.push(toJsonable(message));
      writeProgress(message);
      if (message?.type === "result") {
        resultMessage = message;
        finalResponse = String(message.result ?? "");
      }
    }

    if (!finalResponse) {
      finalResponse = collectAssistantText(messages);
    }

    writePreparedRunArtifact(artifacts[0], finalResponse);
    writePreparedRunArtifact(
      artifacts[1],
      JSON.stringify(toJsonable(messages), null, 2),
    );
    if (request.result_path) {
      writePreparedRunArtifact(
        artifacts[2],
        JSON.stringify(toJsonable(resultMessage ?? {}), null, 2),
      );
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
  }
}

export function writeRunArtifact(request, filePath, content) {
  const artifact = prepareRunArtifact(request, filePath);
  try {
    writePreparedRunArtifact(artifact, content);
  } finally {
    fs.closeSync(artifact.fd);
  }
}

function prepareRunArtifacts(request) {
  const paths = [
    request.child_final_path,
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

export function prepareRunArtifact(request, filePath) {
  const lexicalRunDir = path.resolve(request.run_dir);
  const realRunDir = fs.realpathSync(lexicalRunDir);
  if (lexicalRunDir !== realRunDir) {
    throw new Error(`Run directory must not be a symlink: ${lexicalRunDir}`);
  }
  const resolvedPath = path.resolve(filePath);
  const relativePath = path.relative(lexicalRunDir, resolvedPath);
  if (
    !relativePath ||
    relativePath === ".." ||
    relativePath.startsWith(`..${path.sep}`) ||
    path.isAbsolute(relativePath)
  ) {
    throw new Error(`Bridge artifact must stay inside run_dir: ${filePath}`);
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
    process.stderr.write(
      "Unable to import @anthropic-ai/claude-agent-sdk. Install it with `npm install @anthropic-ai/claude-agent-sdk` in agentic/packages/content_workflow_cli.\n",
    );
    throw error;
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
  delete claudeConfig.env;
  const options = {
    cwd: request.repo_root,
    env: {
      ...process.env,
      CLAUDE_AGENT_SDK_CLIENT_APP: "nvidia-content-workflow-cli/0.1.0",
      ...configEnv,
    },
    ...claudeConfig,
    allowedTools: DEFAULT_TOOLS,
    permissionMode,
    allowDangerouslySkipPermissions: permissionMode === "bypassPermissions",
    persistSession: false,
    sandbox: buildSandboxSettings(request),
    settingSources: [],
    systemPrompt: {
      type: "preset",
      preset: "claude_code",
      append: buildSystemPromptAppend(request),
    },
  };
  // acceptEdits and bypassPermissions auto-approve direct file mutations.
  // Restrict the SDK's actual tool surface in both modes to read-only tools
  // plus Bash. The mandatory OS sandbox confines Bash writes and network.
  if (SANDBOXED_PERMISSION_MODES.has(permissionMode)) {
    options.tools = SANDBOXED_TOOLS;
  } else {
    delete options.tools;
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
  return options;
}

function buildSandboxSettings(request) {
  const allowedDomains = [];
  if (request.workbench_url) {
    let parsed;
    try {
      parsed = new URL(request.workbench_url);
    } catch {
      throw new Error(`Invalid workbench_url: ${request.workbench_url}`);
    }
    if (!["http:", "https:"].includes(parsed.protocol) || !parsed.hostname) {
      throw new Error(`Invalid workbench_url: ${request.workbench_url}`);
    }
    allowedDomains.push(parsed.hostname);
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
    return `${BASE_SYSTEM_PROMPT_APPEND} ${MATERIAL_SYSTEM_PROMPT_APPEND}`;
  }
  return BASE_SYSTEM_PROMPT_APPEND;
}

function filterClaudeConfigEnv(configEnv) {
  const filtered = {};
  const droppedKeys = [];
  for (const [key, value] of Object.entries(configEnv)) {
    if (DANGEROUS_CLAUDE_ENV_KEYS.has(key)) {
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
