#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from "node:fs";
import { createRequire } from "node:module";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

const SECURITY_CRITICAL_CODEX_CONFIG_KEYS = [
  "approval_policy",
  "network_access",
  "permissions",
  "sandbox",
  "sandbox_mode",
  "sandbox_permissions",
  "sandbox_workspace_write",
  "tools",
];
const SUPPORTED_CODEX_CONFIG_KEYS = new Set([
  "model_provider",
  "model_providers",
]);
const DEFAULT_CODEX_SANDBOX_MODE = "workspace-write";
const SUPPORTED_CODEX_AUTH_CREDENTIALS_STORES = new Set([
  "auto",
  "ephemeral",
  "file",
  "keyring",
]);
const SUPPORTED_CODEX_SANDBOX_MODES = new Set([
  DEFAULT_CODEX_SANDBOX_MODE,
]);
const moduleRequire = createRequire(import.meta.url);

async function main() {
  if (process.argv[2] === "--server") {
    const sessionRequestPath = process.argv[3];
    if (!sessionRequestPath) {
      throw new Error("Usage: codex_sdk_bridge.mjs --server <session-request.json>");
    }
    await runServer(sessionRequestPath);
    return;
  }

  const requestPath = process.argv[2];
  if (!requestPath) {
    throw new Error("Usage: codex_sdk_bridge.mjs <request.json>");
  }

  const request = readJsonRequest(requestPath);
  const codexLauncher = prepareCodexLauncher();
  try {
    const { Codex } = await loadCodexSdk();
    const thread = startThread(Codex, request, codexLauncher);
    const finalResponse = await runTurn(thread, request);

    if (finalResponse) {
      process.stdout.write(finalResponse);
      if (!finalResponse.endsWith("\n")) {
        process.stdout.write("\n");
      }
    }
  } finally {
    codexLauncher.cleanup();
  }
}

async function runServer(sessionRequestPath) {
  const sessionRequest = readJsonRequest(sessionRequestPath);
  const codexLauncher = prepareCodexLauncher();
  try {
    const { Codex } = await loadCodexSdk();
    const thread = startThread(Codex, sessionRequest, codexLauncher);
    process.stdout.write(
      JSON.stringify({
        type: "ready",
        schema_version: "content-agents.codex-thread-bridge.v1",
        session_request_path: sessionRequestPath,
      }) + "\n",
    );

    let buffer = "";
    process.stdin.setEncoding("utf8");
    for await (const chunk of process.stdin) {
      buffer += chunk;
      let newlineIndex = buffer.indexOf("\n");
      while (newlineIndex >= 0) {
        const line = buffer.slice(0, newlineIndex).trim();
        buffer = buffer.slice(newlineIndex + 1);
        if (line) {
          const shouldShutdown = await handleServerLine(thread, line);
          if (shouldShutdown) {
            return;
          }
        }
        newlineIndex = buffer.indexOf("\n");
      }
    }
    const finalLine = buffer.trim();
    if (finalLine) {
      await handleServerLine(thread, finalLine);
    }
  } finally {
    codexLauncher.cleanup();
  }
}

async function handleServerLine(thread, line) {
  let message;
  try {
    message = JSON.parse(line);
  } catch (error) {
    process.stdout.write(
      JSON.stringify({
        type: "error",
        error: `Invalid JSON bridge message: ${error.message}`,
      }) + "\n",
    );
    return false;
  }
  if (message.type === "shutdown") {
    process.stdout.write(JSON.stringify({ type: "shutdown_ack" }) + "\n");
    process.exitCode = 0;
    process.stdin.pause();
    return true;
  }
  if (message.type !== "turn" || !message.request_path) {
    process.stdout.write(
      JSON.stringify({
        type: "error",
        request_id: message.request_id ?? null,
        error: "Expected bridge message: {type: 'turn', request_path: string}",
      }) + "\n",
    );
    return false;
  }

  const requestPath = message.request_path;
  try {
    const request = readJsonRequest(requestPath);
    await runTurn(thread, request);
    process.stdout.write(
      JSON.stringify({
        type: "turn_finished",
        request_id: message.request_id ?? null,
        request_path: requestPath,
        returncode: 0,
      }) + "\n",
    );
  } catch (error) {
    process.stdout.write(
      JSON.stringify({
        type: "turn_finished",
        request_id: message.request_id ?? null,
        request_path: requestPath,
        returncode: 1,
        error: String(error.stack ?? error),
      }) + "\n",
    );
  }
  return false;
}

export function startThread(Codex, request, launcher) {
  const config = buildCodexConfig(request);
  const codex = new Codex({
    env: launcher.env,
    config,
    codexPathOverride: launcher.codexPathOverride,
  });
  return codex.startThread(buildThreadOptions(request));
}

export function prepareCodexLauncher(
  sourceEnv = process.env,
  codexExecutable = resolveCodexExecutable(),
) {
  const launcherDir = fs.mkdtempSync(
    path.join(os.tmpdir(), "content-workflow-codex-launcher-"),
  );
  // The wrapper uses ESM imports and top-level await. Keep the explicit .mjs
  // suffix so supported Node 18/20 runtimes do not parse it as CommonJS.
  const launcherPath = path.join(launcherDir, "codex.mjs");
  try {
    fs.chmodSync(launcherDir, 0o700);
    fs.writeFileSync(launcherPath, buildCodexLauncher(codexExecutable), {
      flag: "wx",
      mode: 0o700,
    });
  } catch (error) {
    fs.rmSync(launcherDir, { recursive: true, force: true });
    throw error;
  }
  let cleaned = false;
  return {
    codexPathOverride: launcherPath,
    env: sourceEnv,
    cleanup() {
      if (!cleaned) {
        fs.rmSync(launcherDir, { recursive: true, force: true });
        cleaned = true;
      }
    },
  };
}

export function resolveCodexExecutable() {
  let packageJsonPath;
  try {
    packageJsonPath = moduleRequire.resolve("@openai/codex/package.json");
  } catch (error) {
    throw new Error("Unable to resolve the installed @openai/codex package", {
      cause: error,
    });
  }

  let packageMetadata;
  try {
    packageMetadata = JSON.parse(fs.readFileSync(packageJsonPath, "utf8"));
  } catch (error) {
    throw new Error(`Unable to read Codex package metadata at ${packageJsonPath}`, {
      cause: error,
    });
  }
  const executableEntry =
    typeof packageMetadata.bin === "string"
      ? packageMetadata.bin
      : packageMetadata.bin?.codex;
  if (packageMetadata.name !== "@openai/codex" || !executableEntry) {
    throw new Error(
      `Installed @openai/codex package has no valid codex executable entry: ${packageJsonPath}`,
    );
  }

  const packageRoot = path.dirname(packageJsonPath);
  const lexicalExecutable = path.resolve(packageRoot, executableEntry);
  const lexicalRelative = path.relative(packageRoot, lexicalExecutable);
  if (
    path.isAbsolute(executableEntry) ||
    lexicalRelative === ".." ||
    lexicalRelative.startsWith(`..${path.sep}`)
  ) {
    throw new Error(
      `Installed @openai/codex executable entry escapes its package: ${executableEntry}`,
    );
  }

  let realPackageRoot;
  let realExecutable;
  try {
    realPackageRoot = fs.realpathSync(packageRoot);
    realExecutable = fs.realpathSync(lexicalExecutable);
  } catch (error) {
    throw new Error(`Unable to resolve the Codex executable at ${lexicalExecutable}`, {
      cause: error,
    });
  }
  const realRelative = path.relative(realPackageRoot, realExecutable);
  if (
    realRelative === ".." ||
    realRelative.startsWith(`..${path.sep}`) ||
    path.isAbsolute(realRelative) ||
    !fs.statSync(realExecutable).isFile()
  ) {
    throw new Error(
      `Installed @openai/codex executable is not a package-local file: ${realExecutable}`,
    );
  }
  try {
    fs.accessSync(realExecutable, fs.constants.X_OK);
  } catch (error) {
    throw new Error(`Installed Codex entry is not executable: ${realExecutable}`, {
      cause: error,
    });
  }
  return realExecutable;
}

function buildCodexLauncher(codexExecutable) {
  return `#!${process.execPath}\n` +
    `import { spawn } from "node:child_process";\n` +
    `import process from "node:process";\n` +
    `const args = process.argv.slice(2);\n` +
    `if (args[0] !== "exec" || args[1] !== "--experimental-json") {\n` +
    `  process.stderr.write("Codex SDK launcher accepts only exec --experimental-json\\n");\n` +
    `  process.exit(2);\n` +
    `}\n` +
    `const forwarded = ["exec", "--ignore-user-config", "--ignore-rules", ...args.slice(1)];\n` +
    `const child = spawn(${JSON.stringify(codexExecutable)}, forwarded, {\n` +
    `  stdio: ["pipe", "inherit", "inherit"],\n` +
    `  env: process.env,\n` +
    `});\n` +
    `process.stdin.pipe(child.stdin);\n` +
    `const forwardedSignals = ["SIGINT", "SIGTERM", "SIGHUP"];\n` +
    `const signalHandlers = new Map();\n` +
    `for (const signal of forwardedSignals) {\n` +
    `  const handler = () => {\n` +
    `    if (!child.killed) child.kill(signal);\n` +
    `  };\n` +
    `  signalHandlers.set(signal, handler);\n` +
    `  process.on(signal, handler);\n` +
    `}\n` +
    `const result = await new Promise((resolve, reject) => {\n` +
    `  child.once("error", reject);\n` +
    `  child.once("exit", (code, signal) => resolve({ code, signal }));\n` +
    `});\n` +
    `if (result.signal) {\n` +
    `  for (const [signal, handler] of signalHandlers) {\n` +
    `    process.removeListener(signal, handler);\n` +
    `  }\n` +
    `  process.kill(process.pid, result.signal);\n` +
    `} else {\n` +
    `  process.exit(result.code ?? 1);\n` +
    `}\n`;
}

export function buildThreadOptions(request) {
  // Codex owns model and effort at thread scope; persistent refinement turns
  // resume this thread and inherit both settings.
  const options = {
    workingDirectory: request.repo_root,
    skipGitRepoCheck: true,
  };
  if (request.model) {
    options.model = request.model;
  }
  if (request.model_reasoning_effort) {
    options.modelReasoningEffort = request.model_reasoning_effort;
  }
  return options;
}

async function runTurn(thread, request) {
  const input = buildInput(request);
  const turnOptions = buildTurnOptions(request);
  const artifacts = prepareRunArtifacts(request);
  try {
    const turn =
      Object.keys(turnOptions).length > 0
        ? await thread.run(input, turnOptions)
        : await thread.run(input);
    const finalResponse = String(turn.finalResponse ?? turn.final_response ?? "");

    writePreparedRunArtifact(artifacts[0], finalResponse);
    writePreparedRunArtifact(
      artifacts[1],
      JSON.stringify(toJsonable(turn.items ?? []), null, 2),
    );
    if (request.result_path) {
      writePreparedRunArtifact(
        artifacts[2],
        JSON.stringify(toJsonable(turn), null, 2),
      );
    }
    return finalResponse;
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
    // Unlinking first is safe for symlinks and hard links: it removes only the
    // directory entry, then O_EXCL creates a new private regular file.
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
  // The descriptor was opened before the child turn. Even if a surviving
  // descendant races the path checks, all writes remain bound to this inode.
  assertPreparedRunArtifact(artifact);
  fs.ftruncateSync(artifact.fd, 0);
  fs.writeFileSync(artifact.fd, content, "utf8");
  fs.fchmodSync(artifact.fd, 0o600);
  fs.fsyncSync(artifact.fd);
  assertPreparedRunArtifact(artifact);
}

function readJsonRequest(requestPath) {
  try {
    return JSON.parse(fs.readFileSync(requestPath, "utf8"));
  } catch (error) {
    throw new Error(
      `Invalid Codex SDK bridge request file at ${requestPath}: ${error.message}`,
    );
  }
}

async function loadCodexSdk() {
  try {
    return await import("@openai/codex-sdk");
  } catch (error) {
    process.stderr.write(
      "Unable to import @openai/codex-sdk. Install it with `npm install @openai/codex-sdk` in agentic/packages/content_workflow_cli or the repository root.\n",
    );
    throw error;
  }
}

export function buildCodexConfig(request) {
  const codexConfig = { ...(request.codex_config ?? {}) };
  dropSecurityCriticalCodexConfigKeys(codexConfig);
  const sandboxMode = resolveCodexSandboxMode(request.codex_sandbox_mode);
  const credentialsStore = resolveCodexAuthCredentialsStore(
    request.cli_auth_credentials_store,
  );
  // `codex_config` is a trusted local escape hatch forwarded to the Codex SDK
  // for provider/auth customization. Keep unattended execution controls fixed
  // after the spread so request config cannot relax the wrapper sandbox policy.
  const config = {
    ...codexConfig,
    approval_policy: "never",
    sandbox_mode: sandboxMode,
    features: {
      plugins: false,
    },
  };
  if (credentialsStore !== null) {
    config.cli_auth_credentials_store = credentialsStore;
  }
  if (sandboxMode === DEFAULT_CODEX_SANDBOX_MODE) {
    // Content Workbench is an HTTP service. Keep filesystem confinement while
    // restoring the network access required by every workflow.
    config.sandbox_workspace_write = {
      network_access: true,
      exclude_tmpdir_env_var: true,
      exclude_slash_tmp: true,
    };
  }
  if (request.codex_base_url) {
    validateCodexBaseUrl(request.codex_base_url);
    config.openai_base_url = request.codex_base_url;
  }
  return config;
}

function resolveCodexAuthCredentialsStore(value) {
  if (!SUPPORTED_CODEX_AUTH_CREDENTIALS_STORES.has(value)) {
    return null;
  }
  return value;
}

function validateCodexBaseUrl(value) {
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error(`Invalid codex_base_url: ${value}`);
  }
  if (!["http:", "https:"].includes(parsed.protocol) || !parsed.host) {
    throw new Error(`Invalid codex_base_url: ${value}`);
  }
}

function resolveCodexSandboxMode(value) {
  if (value === undefined || value === null || value === "") {
    return DEFAULT_CODEX_SANDBOX_MODE;
  }
  if (!SUPPORTED_CODEX_SANDBOX_MODES.has(value)) {
    throw new Error(
      `Unsupported Codex sandbox mode: ${value}. Expected one of: ${Array.from(
        SUPPORTED_CODEX_SANDBOX_MODES,
      ).join(", ")}`,
    );
  }
  return value;
}

function dropSecurityCriticalCodexConfigKeys(codexConfig) {
  const droppedKeys = Object.keys(codexConfig).filter(
    (key) =>
      !SUPPORTED_CODEX_CONFIG_KEYS.has(key) ||
      SECURITY_CRITICAL_CODEX_CONFIG_KEYS.some(
        (criticalKey) => key.replaceAll(/["']/g, "").includes(criticalKey),
      ),
  );
  if (droppedKeys.length === 0) {
    return;
  }
  for (const key of droppedKeys) {
    delete codexConfig[key];
  }
  process.stderr.write(
    `Ignoring security-critical Codex config key(s): ${droppedKeys.join(", ")}.\n`,
  );
}

export function buildTurnOptions(request) {
  const options = {};
  if (request.output_schema) {
    options.outputSchema = request.output_schema;
  }
  return options;
}

function buildInput(request) {
  const input = [{ type: "text", text: request.prompt }];
  for (const imagePath of request.reference_images ?? []) {
    validateReadableFile(imagePath, "reference image");
    input.push({ type: "text", text: `Reference image: ${path.basename(imagePath)}` });
    input.push({ type: "local_image", path: imagePath });
  }
  for (const filePath of request.reference_files ?? []) {
    validateReadableFile(filePath, "reference file");
    input.push({
      type: "text",
      text: `Reference file: ${path.basename(filePath)} (${filePath})`,
    });
  }
  for (const image of request.prompt_image_inputs ?? []) {
    const imagePath = image?.path;
    if (!imagePath) {
      continue;
    }
    const label = image?.label ?? "Prompt image";
    validateReadableFile(imagePath, label);
    input.push({ type: "text", text: `${label}: ${path.basename(imagePath)}` });
    input.push({ type: "local_image", path: imagePath });
  }
  return input;
}

function validateReadableFile(filePath, label) {
  let stat;
  try {
    fs.accessSync(filePath, fs.constants.R_OK);
    stat = fs.statSync(filePath);
  } catch (error) {
    throw new Error(`Unable to read ${label}: ${filePath}`, { cause: error });
  }
  if (!stat.isFile()) {
    throw new Error(`${label} is not a file: ${filePath}`);
  }
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
