#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import fs from "node:fs";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import tls from "node:tls";
import { fileURLToPath, pathToFileURL } from "node:url";

const SECURITY_CRITICAL_CODEX_CONFIG_KEYS = [
  "approval_policy",
  "env",
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
const DANGER_FULL_ACCESS_CODEX_SANDBOX_MODE = "danger-full-access";
const CONTENT_WORKFLOW_CODEX_PERMISSION_PROFILE = "content-workflow-child";
const CONTENT_WORKFLOW_CUSTOM_MODEL_PROVIDER = "content_workflow_custom";
const SUPPORTED_CODEX_AUTH_CREDENTIALS_STORES = new Set([
  "auto",
  "ephemeral",
  "file",
  "keyring",
]);
const SUPPORTED_CODEX_SANDBOX_MODES = new Set([
  DEFAULT_CODEX_SANDBOX_MODE,
  DANGER_FULL_ACCESS_CODEX_SANDBOX_MODE,
]);
const REASONING_ADAPTER_HOSTS = new Set(["127.0.0.1", "[::1]"]);
export const SUPPORTED_CODEX_SDK_VERSION = "0.147.0";
const SUPPORTED_STREAM_EVENT_TYPES = new Set([
  "thread.started",
  "turn.started",
  "turn.completed",
  "turn.failed",
  "turn.cancelled",
  "item.started",
  "item.updated",
  "item.completed",
  "error",
]);
const moduleRequire = createRequire(import.meta.url);
const BRIDGE_PATH = fileURLToPath(import.meta.url);
const PYTHON_PROGRAM = /^(?:py|python(?:\d+(?:\.\d+)*)?|pypy(?:\d+)?)(?:\.exe)?$/i;
const WINDOWS_SCRIPT_PROGRAM = /\.(?:bat|cmd|ps1|vbs|wsf)$/i;
const WINDOWS_SCRIPT_HOST_PROGRAMS = new Set([
  "cscript",
  "cscript.exe",
  "mshta",
  "mshta.exe",
  "wscript",
  "wscript.exe",
]);
const SHELL_PROGRAMS = new Set(["bash", "dash", "sh", "zsh"]);
const POWERSHELL_PROGRAMS = new Set([
  "powershell",
  "powershell.exe",
  "pwsh",
  "pwsh.exe",
]);
const POWERSHELL_OPTIONS_WITH_VALUES = new Set([
  "-executionpolicy",
  "-inputformat",
  "-outputformat",
  "-version",
  "-windowstyle",
]);
const POWERSHELL_OPTIONS_WITHOUT_VALUES = new Set([
  "-interactive",
  "-mta",
  "-noexit",
  "-nologo",
  "-noninteractive",
  "-noprofile",
  "-noprofileloadtime",
  "-sta",
]);
const POWERSHELL_INDIRECT_PROCESS_LAUNCHERS = new Set([
  "cmd",
  "cmd.exe",
  "icm",
  "iex",
  "invoke-command",
  "invoke-expression",
  "saps",
  "start",
  "start-job",
  "start-process",
  "start-threadjob",
]);
const POWERSHELL_PROCESS_CREATION_PROGRAMS = new Set([
  "ii",
  "iwmi",
  "invoke-cimmethod",
  "invoke-item",
  "invoke-wmimethod",
]);
const POWERSHELL_COMMAND_DEFINITION_PROGRAMS = new Set([
  "add-type",
  "filter",
  "function",
  "import-alias",
  "import-module",
  "ipal",
  "ipmo",
  "nal",
  "new-alias",
  "new-module",
  "nmo",
  "sal",
  "set-alias",
  "workflow",
]);
const POWERSHELL_PROVIDER_MUTATION_PROGRAMS = new Set([
  "ac",
  "add-content",
  "copy",
  "copy-item",
  "cp",
  "cpi",
  "mi",
  "move",
  "move-item",
  "mv",
  "new-item",
  "ni",
  "ren",
  "rename-item",
  "rni",
  "sc",
  "set-content",
  "set-item",
  "si",
]);
const COMMAND_WRAPPERS = new Set(["command", "env", "nohup"]);
const INDIRECT_PROCESS_LAUNCHERS = new Set([
  "builtin",
  "cmd",
  "cmd.exe",
  "eval",
  "exec",
  "parallel",
  "py.test",
  "pytest",
  "sudo",
  "xargs",
]);
const FIND_EXEC_OPTIONS = new Set(["-exec", "-execdir", "-ok", "-okdir"]);
const UV_RUN_PYTHON_MODULE_OPTIONS = new Set(["-m", "--module"]);
const UV_RUN_PYTHON_SCRIPT_OPTIONS = new Set(["-s", "--script", "--gui-script"]);
const UV_RUN_OPTIONS_WITH_VALUES = new Set([
  "--allow-insecure-host",
  "--cache-dir",
  "--color",
  "--config-file",
  "--config-setting",
  "--config-settings-package",
  "--default-index",
  "--directory",
  "--env-file",
  "--exclude-newer",
  "--exclude-newer-package",
  "--extra",
  "--extra-index-url",
  "--find-links",
  "--fork-strategy",
  "--group",
  "--index",
  "--index-strategy",
  "--index-url",
  "--keyring-provider",
  "--link-mode",
  "--no-binary-package",
  "--no-build-isolation-package",
  "--no-build-package",
  "--no-editable-package",
  "--no-extra",
  "--no-group",
  "--no-sources-package",
  "--only-group",
  "--package",
  "--prerelease",
  "--project",
  "--python",
  "--python-platform",
  "--refresh-package",
  "--reinstall-package",
  "--resolution",
  "--upgrade-group",
  "--upgrade-package",
  "--with",
  "--with-editable",
  "--with-requirements",
  "-C",
  "-P",
  "-f",
  "-i",
  "-p",
  "-w",
]);
const UV_RUN_OPTIONS_WITHOUT_VALUES = new Set([
  "--active",
  "--all-extras",
  "--all-groups",
  "--all-packages",
  "--compile-bytecode",
  "--exact",
  "--frozen",
  "--help",
  "--isolated",
  "--locked",
  "--managed-python",
  "--no-binary",
  "--no-build",
  "--no-build-isolation",
  "--no-cache",
  "--no-config",
  "--no-default-groups",
  "--no-dev",
  "--no-editable",
  "--no-env-file",
  "--no-index",
  "--no-managed-python",
  "--no-progress",
  "--no-project",
  "--no-python-downloads",
  "--no-sources",
  "--no-sync",
  "--offline",
  "--only-dev",
  "--quiet",
  "--refresh",
  "--reinstall",
  "--system-certs",
  "--upgrade",
  "--verbose",
  "-U",
  "-h",
  "-n",
  "-q",
  "-v",
]);

function shellQuote(value) {
  return `'${String(value).replaceAll("'", `'"'"'`)}'`;
}

function sandboxSmokeWriteCommand(marker, contents) {
  if (process.platform === "win32") {
    const quote = (value) => String(value).replaceAll("'", "''");
    return `Set-Content -NoNewline -LiteralPath '${quote(marker)}' -Value '${quote(contents)}'`;
  }
  return `printf ${shellQuote(contents)} > ${shellQuote(marker)}`;
}

export function sandboxSmokeHostExecutables(
  platform = process.platform,
  environment = process.env,
  resolvePath = fs.realpathSync,
  isFile = (candidate) => fs.statSync(candidate).isFile(),
) {
  if (platform !== "win32") return [];
  const systemRoot = environment.SystemRoot;
  if (!systemRoot || !path.win32.isAbsolute(systemRoot)) {
    throw new Error("Windows Codex sandbox smoke requires an absolute SystemRoot");
  }
  const powershell = resolvePath(
    path.win32.join(
      systemRoot,
      "System32",
      "WindowsPowerShell",
      "v1.0",
      "powershell.exe",
    ),
  );
  if (!isFile(powershell)) {
    throw new Error("Windows Codex sandbox smoke requires system PowerShell");
  }
  return [{ name: "powershell", paths: [powershell] }];
}

function readExactSandboxSmokeMarker(marker, expected) {
  const expectedBytes = Buffer.from(expected, "utf8");
  let descriptor;
  try {
    const before = fs.lstatSync(marker);
    if (!before.isFile() || before.size !== expectedBytes.length) {
      return false;
    }
    descriptor = fs.openSync(
      marker,
      fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW ?? 0),
    );
    const after = fs.fstatSync(descriptor);
    if (
      !after.isFile() ||
      after.size !== expectedBytes.length ||
      after.dev !== before.dev ||
      after.ino !== before.ino
    ) {
      return false;
    }
    const actual = Buffer.alloc(expectedBytes.length + 1);
    return (
      fs.readSync(descriptor, actual, 0, actual.length, 0) === expectedBytes.length &&
      actual.subarray(0, expectedBytes.length).equals(expectedBytes)
    );
  } catch {
    return false;
  } finally {
    if (descriptor !== undefined) {
      fs.closeSync(descriptor);
    }
  }
}

export function cleanupSandboxSmokeArtifacts(artifacts) {
  for (const artifact of artifacts) {
    try {
      fs.unlinkSync(artifact);
    } catch (error) {
      if (error?.code !== "ENOENT") {
        process.stderr.write(`Unable to remove Codex sandbox smoke artifact: ${error.message}\n`);
      }
    }
  }
}

function shellTokens(command, preserveBackslashes = false) {
  const tokens = [];
  let token = "";
  let quote = null;
  let escaped = false;
  const flush = () => {
    if (token) {
      tokens.push(token);
      token = "";
    }
  };
  for (let index = 0; index < command.length; index += 1) {
    const character = command[index];
    if (escaped) {
      token += character;
      escaped = false;
      continue;
    }
    if (
      character === "\\" &&
      quote !== "'" &&
      process.platform !== "win32" &&
      !preserveBackslashes
    ) {
      if (
        quote === '"' &&
        !["\\", "$", "`", '"', "\n"].includes(command[index + 1])
      ) {
        token += character;
        continue;
      }
      escaped = true;
      continue;
    }
    if (quote) {
      if (character === quote) {
        quote = null;
      } else {
        token += character;
      }
      continue;
    }
    if (character === "'" || character === '"') {
      quote = character;
      continue;
    }
    if (
      preserveBackslashes &&
      (character === "{" || character === "}") &&
      !(character === "{" && ["@", "$"].includes(command[index - 1]))
    ) {
      flush();
      tokens.push(character);
      continue;
    }
    if (";&|\n".includes(character)) {
      flush();
      const doubled = character !== "\n" && command[index + 1] === character;
      tokens.push(doubled ? character + character : character);
      if (doubled) index += 1;
      continue;
    }
    if (/\s/.test(character)) {
      flush();
      continue;
    }
    token += character;
  }
  if (escaped) token += "\\";
  flush();
  return tokens;
}

function hasUnsupportedShellGrouping(command, powershellSyntax = false) {
  let quote = null;
  let escaped = false;
  const isBoundary = (value) => value === undefined || /[\s;&|]/.test(value);
  for (let index = 0; index < command.length; index += 1) {
    const character = command[index];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (character === "\\" && quote !== "'" && process.platform !== "win32") {
      escaped = true;
      continue;
    }
    if (quote) {
      if (character === quote) quote = null;
      continue;
    }
    if (character === "'" || character === '"') {
      quote = character;
      continue;
    }
    if (character === "(" || character === ")") return true;
    if (
      !powershellSyntax &&
      (character === "{" || character === "}") &&
      isBoundary(command[index - 1]) &&
      isBoundary(command[index + 1])
    ) {
      return true;
    }
  }
  return false;
}

function isLiteralApplyPatchCommand(command, toolName) {
  const normalized = command.replaceAll("\r\n", "\n");
  const lines = normalized.split("\n");
  while (lines.at(-1) === "") lines.pop();
  if (
    toolName === "apply_patch" &&
    lines.length >= 3 &&
    lines[0] === "*** Begin Patch" &&
    lines.at(-1) === "*** End Patch"
  ) {
    return true;
  }
  const opening = lines[0]?.match(/^apply_patch[ \t]+<<'([A-Za-z_][A-Za-z0-9_]*)'[ \t]*$/);
  if (!opening) return false;
  const delimiter = opening[1];
  return (
    lines.length >= 4 &&
    lines[1] === "*** Begin Patch" &&
    !lines.slice(1, -1).includes(delimiter) &&
    lines.at(-2) === "*** End Patch" &&
    lines.at(-1) === delimiter
  );
}

function shellSegments(command, preserveBackslashes = false) {
  const segments = [];
  let segment = [];
  for (const token of shellTokens(command, preserveBackslashes)) {
    if (
      [";", "&&", "||", "|", "\n"].includes(token) ||
      (preserveBackslashes && ["{", "}"].includes(token))
    ) {
      if (segment.length) segments.push(segment);
      segment = [];
    } else {
      segment.push(token);
    }
  }
  if (segment.length) segments.push(segment);
  return segments;
}

function programName(value) {
  return path.basename(String(value)).toLowerCase();
}

function powershellProgramName(value) {
  return String(value).split(/[\\/]/).at(-1).toLowerCase();
}

function firstProgramIndex(segment) {
  let index = 0;
  while (
    index < segment.length &&
    (["!", "do", "if", "then", "until", "while"].includes(segment[index]) ||
      /^[A-Za-z_][A-Za-z0-9_]*=.*/.test(segment[index]))
  ) {
    index += 1;
  }
  while (index < segment.length && COMMAND_WRAPPERS.has(programName(segment[index]))) {
    const wrapper = programName(segment[index]);
    index += 1;
    if (wrapper === "env") {
      while (index < segment.length) {
        const token = segment[index];
        if (/^[A-Za-z_][A-Za-z0-9_]*=.*/.test(token)) {
          index += 1;
          continue;
        }
        if (token === "--") {
          index += 1;
          break;
        }
        if (["-S", "--split-string"].includes(token) || /^(?:--split-string=|-S).+/.test(token)) {
          return -1;
        }
        if (["-u", "--unset", "-C", "--chdir", "-a", "--argv0"].includes(token)) {
          if (index + 1 >= segment.length) return -1;
          index += 2;
          continue;
        }
        if (/^(?:--unset|--chdir|--argv0)=.+/.test(token) || /^-[uCa].+/.test(token)) {
          index += 1;
          continue;
        }
        if (["-i", "--ignore-environment", "-0", "--null", "--debug", "--help", "--version"].includes(token)) {
          index += 1;
          continue;
        }
        if (token.startsWith("-")) return -1;
        break;
      }
    } else {
      while (index < segment.length && segment[index].startsWith("-")) index += 1;
    }
  }
  while (["nice", "timeout"].includes(programName(segment[index] ?? ""))) {
    const wrapper = programName(segment[index]);
    index += 1;
    if (wrapper === "nice") {
      while (index < segment.length) {
        const token = segment[index];
        if (token === "--") {
          index += 1;
          break;
        }
        if (["-n", "--adjustment"].includes(token)) {
          if (index + 1 >= segment.length) return -1;
          index += 2;
          continue;
        }
        if (/^(?:--adjustment=|-n).+/.test(token) || /^-\d+$/.test(token)) {
          index += 1;
          continue;
        }
        if (["--help", "--version"].includes(token)) {
          index += 1;
          continue;
        }
        if (token.startsWith("-")) return -1;
        break;
      }
      continue;
    }

    while (index < segment.length) {
      const token = segment[index];
      if (token === "--") {
        index += 1;
        break;
      }
      if (["-k", "--kill-after", "-s", "--signal"].includes(token)) {
        if (index + 1 >= segment.length) return -1;
        index += 2;
        continue;
      }
      if (/^(?:--kill-after|--signal)=.+/.test(token) || /^-[ks].+/.test(token)) {
        index += 1;
        continue;
      }
      if (["--preserve-status", "--foreground", "--verbose", "--help", "--version"].includes(token)) {
        index += 1;
        continue;
      }
      if (token.startsWith("-")) return -1;
      break;
    }
    if (index >= segment.length) return segment.length;
    index += 1;
  }
  if (index < segment.length && COMMAND_WRAPPERS.has(programName(segment[index]))) {
    const nestedIndex = firstProgramIndex(segment.slice(index));
    return nestedIndex < 0 ? -1 : index + nestedIndex;
  }
  return index;
}

function powershellCommandDefinesOrImportsCommand(command) {
  return shellSegments(command, true).some((segment) => {
    if (!segment.length) return false;
    const program = powershellProgramName(segment[0]);
    const operands = segment.slice(1).map(String);
    if (POWERSHELL_COMMAND_DEFINITION_PROGRAMS.has(program)) return true;
    if (program === "using" && operands[0]?.toLowerCase() === "module") {
      return true;
    }
    return (
      POWERSHELL_PROVIDER_MUTATION_PROGRAMS.has(program) &&
      operands.some((value) => /(?:^|[\\/])(?:alias|function):/i.test(value))
    );
  });
}

function powershellActiveCode(command) {
  const tokenCharacter = (value) => /[A-Za-z0-9_.\\/-]/.test(value ?? "");
  let activeCode = "";
  let index = 0;
  while (index < command.length) {
    const character = command[index];
    const atCommentBoundary =
      index === 0 || /[\s;|&(){}=,+*/%-]/.test(command[index - 1]);
    if (character === "#" && atCommentBoundary) {
      const newline = command.indexOf("\n", index + 1);
      if (newline < 0) break;
      activeCode += "\n";
      index = newline + 1;
      continue;
    }
    if (character === "<" && command[index + 1] === "#" && atCommentBoundary) {
      const commentEnd = command.indexOf("#>", index + 2);
      activeCode += " ";
      index = commentEnd < 0 ? command.length : commentEnd + 2;
      continue;
    }
    if (character === "'" || character === '"') {
      const quote = character;
      let literal = "";
      let hasSubexpression = false;
      let quoteEnd = index + 1;
      while (quoteEnd < command.length) {
        if (quote === "'" && command.slice(quoteEnd, quoteEnd + 2) === "''") {
          literal += "'";
          quoteEnd += 2;
          continue;
        }
        if (command[quoteEnd] === quote) break;
        if (quote === '"' && command.slice(quoteEnd, quoteEnd + 2) === "$(") {
          hasSubexpression = true;
        }
        if (quote === '"' && command[quoteEnd] === "`" && quoteEnd + 1 < command.length) {
          const escaped = command[quoteEnd + 1];
          if (escaped !== "\r" && escaped !== "\n") literal += escaped;
          quoteEnd += escaped === "\r" && command[quoteEnd + 2] === "\n" ? 3 : 2;
          continue;
        }
        literal += command[quoteEnd];
        quoteEnd += 1;
      }
      const nextIndex = Math.min(quoteEnd + 1, command.length);
      const attached =
        tokenCharacter(command[index - 1]) || tokenCharacter(activeCode.at(-1));
      const projected = hasSubexpression ? powershellActiveCode(literal) : literal;
      activeCode += attached || hasSubexpression ? projected : " ";
      index = nextIndex;
      continue;
    }
    if (character === "`" && index + 1 < command.length) {
      const escaped = command[index + 1];
      if (escaped !== "\r" && escaped !== "\n") activeCode += escaped;
      index += escaped === "\r" && command[index + 2] === "\n" ? 3 : 2;
      continue;
    }
    activeCode += character;
    index += 1;
  }
  return activeCode;
}

function powershellCommandCreatesProcess(command) {
  const programs = [...POWERSHELL_PROCESS_CREATION_PROGRAMS].join("|");
  // Unquoted mentions fail closed; only inert strings and comments are masked.
  return new RegExp(
    `(?:^|[^A-Za-z0-9_-])(?:[A-Za-z0-9_.-]+[\\\\/])?` +
      `(?:${programs})(?=$|[\\s;|&(){}>,])`,
    "i",
  ).test(powershellActiveCode(command));
}

function resolveCommandPath(value, cwd) {
  if (
    !value.includes("/") &&
    !value.includes("\\") &&
    !value.toLowerCase().endsWith(".py") &&
    !WINDOWS_SCRIPT_PROGRAM.test(value)
  ) return null;
  return path.resolve(cwd, value);
}

function verifiedCheckedInScript(scriptPath, trustedRepoRoot) {
  if (!trustedRepoRoot || !path.isAbsolute(trustedRepoRoot)) return false;
  let root;
  let resolvedScript;
  try {
    root = fs.realpathSync(trustedRepoRoot);
    const metadata = fs.lstatSync(scriptPath);
    if (!metadata.isFile() || metadata.isSymbolicLink()) return false;
    resolvedScript = fs.realpathSync(scriptPath);
  } catch {
    return false;
  }
  const relative = path.relative(root, resolvedScript);
  if (!relative || relative === ".." || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
    return false;
  }
  const expected = spawnSync(
    "git",
    ["-C", root, "rev-parse", "--verify", `HEAD:${relative.split(path.sep).join("/")}`],
    { encoding: "utf8" },
  );
  if (expected.status !== 0) return false;
  const actual = spawnSync(
    "git",
    ["-C", root, "hash-object", "--no-filters", "--", resolvedScript],
    { encoding: "utf8" },
  );
  if (actual.status === 0 && actual.stdout.trim() === expected.stdout.trim()) {
    return true;
  }
  if (process.platform !== "win32") return false;
  let contents;
  try {
    const metadata = fs.statSync(resolvedScript);
    if (metadata.size > 16 * 1024 * 1024) return false;
    contents = fs.readFileSync(resolvedScript);
  } catch {
    return false;
  }
  const normalized = Buffer.allocUnsafe(contents.length);
  let normalizedLength = 0;
  let sawCrlf = false;
  for (let index = 0; index < contents.length; index += 1) {
    if (contents[index] === 13 && contents[index + 1] === 10) {
      sawCrlf = true;
      continue;
    }
    normalized[normalizedLength] = contents[index];
    normalizedLength += 1;
  }
  if (!sawCrlf) return false;
  // Git commonly stores LF blobs while core.autocrlf checks an unchanged
  // script out with CRLF on Windows. Normalize only that built-in newline
  // representation; never invoke repository-defined clean filters here.
  const normalizedHash = spawnSync(
    "git",
    ["-C", root, "hash-object", "--stdin"],
    { encoding: "utf8", input: normalized.subarray(0, normalizedLength) },
  );
  return (
    normalizedHash.status === 0 &&
    normalizedHash.stdout.trim() === expected.stdout.trim()
  );
}

function verifiedStagedScript(scriptPath, trustedScriptDigests, childCwd) {
  if (
    !trustedScriptDigests ||
    typeof trustedScriptDigests !== "object" ||
    Array.isArray(trustedScriptDigests)
  ) {
    return false;
  }
  let resolvedScript;
  try {
    const metadata = fs.lstatSync(scriptPath);
    if (!metadata.isFile() || metadata.isSymbolicLink()) return false;
    resolvedScript = fs.realpathSync(scriptPath);
    const resolvedChildCwd = fs.realpathSync(childCwd);
    const childRelative = path.relative(resolvedChildCwd, resolvedScript);
    // A digest check before execution is not an immutability boundary when the
    // child can rewrite the same path in the shell invocation being approved.
    // Digest-bound helpers must live outside the child-writable workspace.
    if (
      !childRelative ||
      (childRelative !== ".." &&
        !childRelative.startsWith(`..${path.sep}`) &&
        !path.isAbsolute(childRelative))
    ) {
      return false;
    }
  } catch {
    return false;
  }
  const expected = trustedScriptDigests[resolvedScript];
  if (typeof expected !== "string" || !/^[0-9a-f]{64}$/.test(expected)) {
    return false;
  }
  const trustedEntries = Object.entries(trustedScriptDigests);
  if (
    trustedEntries.length === 0 ||
    trustedEntries.some(
      ([candidate, digest]) =>
        !path.isAbsolute(candidate) ||
        typeof digest !== "string" ||
        !/^[0-9a-f]{64}$/.test(digest),
    )
  ) {
    return false;
  }
  let stagedRoot = path.dirname(resolvedScript);
  for (const [candidate] of trustedEntries) {
    let relative = path.relative(stagedRoot, candidate);
    while (
      stagedRoot !== path.dirname(stagedRoot) &&
      (relative === ".." || relative.startsWith(`..${path.sep}`))
    ) {
      stagedRoot = path.dirname(stagedRoot);
      relative = path.relative(stagedRoot, candidate);
    }
  }
  if (stagedRoot === path.parse(stagedRoot).root) return false;
  const trustedPaths = new Set(trustedEntries.map(([candidate]) => candidate));
  const observedPaths = new Set();
  const pendingDirectories = [stagedRoot];
  try {
    while (pendingDirectories.length) {
      const directory = pendingDirectories.pop();
      for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
        const candidate = path.join(directory, entry.name);
        if (entry.isSymbolicLink()) return false;
        if (entry.isDirectory()) {
          pendingDirectories.push(candidate);
        } else if (entry.isFile() && entry.name.endsWith(".py")) {
          observedPaths.add(fs.realpathSync(candidate));
        }
      }
    }
  } catch {
    return false;
  }
  if (
    observedPaths.size !== trustedPaths.size ||
    ![...observedPaths].every((candidate) => trustedPaths.has(candidate))
  ) {
    return false;
  }
  return trustedEntries.every(([candidate, digest]) => {
    try {
      const metadata = fs.lstatSync(candidate);
      return (
        metadata.isFile() &&
        !metadata.isSymbolicLink() &&
        fs.realpathSync(candidate) === candidate &&
        createHash("sha256").update(fs.readFileSync(candidate)).digest("hex") === digest
      );
    } catch {
      return false;
    }
  });
}

function verifiedTrustedScript(
  scriptPath,
  trustedRepoRoot,
  trustedScriptDigests,
  childCwd,
) {
  return (
    verifiedCheckedInScript(scriptPath, trustedRepoRoot) ||
    verifiedStagedScript(scriptPath, trustedScriptDigests, childCwd)
  );
}

function pythonScriptDecision(script, cwd, trustedRepoRoot, trustedScriptDigests) {
  const resolved = path.resolve(cwd, script);
  if (
    resolved &&
    verifiedTrustedScript(resolved, trustedRepoRoot, trustedScriptDigests, cwd)
  ) return null;
  return `Python script is not a provenance-verified, unchanged Git file: ${script}`;
}

function shellScriptDecision(script, cwd, trustedRepoRoot, trustedScriptDigests) {
  const resolved = path.resolve(cwd, script);
  if (
    resolved &&
    verifiedTrustedScript(resolved, trustedRepoRoot, trustedScriptDigests, cwd)
  ) return null;
  return `Shell script is not a provenance-verified, unchanged Git file: ${script}`;
}

function inspectShell(args, cwd, trustedRepoRoot, trustedScriptDigests) {
  const commandIndex = args.findIndex(
    (value) => /^-[^-]*c[^-]*$/.test(value),
  );
  if (commandIndex >= 0) {
    if (!args[commandIndex + 1]) {
      return "Shell command execution is missing its command operand.";
    }
    return evaluateModelPythonCommand(
      args[commandIndex + 1],
      cwd,
      trustedRepoRoot,
      trustedScriptDigests,
    );
  }
  if (args.some((value) => value.includes("$(") || value.includes("`"))) {
    return "Shell command substitutions are not permitted in child-agent turns.";
  }

  let index = 0;
  while (index < args.length && args[index].startsWith("-")) {
    const option = args[index];
    if (option === "--") {
      index += 1;
      break;
    }
    if (option === "-" || /^-[^-]*s/.test(option)) {
      return "Shell stdin execution is not permitted in child-agent turns.";
    }
    index += 1;
  }
  if (index >= args.length) {
    return "Shell stdin execution is not permitted in child-agent turns.";
  }
  const script = args[index];
  if (/[$`]/.test(script)) {
    return `Dynamically resolved shell scripts are not permitted in child-agent turns: ${script}`;
  }
  return shellScriptDecision(script, cwd, trustedRepoRoot, trustedScriptDigests);
}

function inspectPowerShell(args, cwd, trustedRepoRoot, trustedScriptDigests) {
  let index = 0;
  while (index < args.length) {
    const rawOption = String(args[index]);
    const option = rawOption.toLowerCase();
    if (!option.startsWith("-")) {
      return `Positional PowerShell execution is not permitted in child-agent turns: ${rawOption}`;
    }

    const separatorIndex = option.indexOf(":");
    const optionName = separatorIndex >= 0 ? option.slice(0, separatorIndex) : option;
    const attachedValue =
      separatorIndex >= 0 ? rawOption.slice(separatorIndex + 1) : null;
    const parameterName = optionName.replace(/^-+/, "");

    // powershell.exe accepts unambiguous parameter prefixes. Match those
    // prefixes before the exact safe-option allowlist so forms such as -Comm,
    // -ec, and -f cannot bypass the execution policy.
    if ("encodedcommand".startsWith(parameterName) || parameterName === "ec") {
      return "Encoded PowerShell commands are not permitted in child-agent turns.";
    }
    if ("encodedarguments".startsWith(parameterName)) {
      return "Encoded PowerShell arguments are not permitted in child-agent turns.";
    }
    if ("file".startsWith(parameterName)) {
      return "PowerShell file execution is not permitted in child-agent turns.";
    }
    if ("commandwithargs".startsWith(parameterName) && parameterName.length > 7) {
      return "PowerShell CommandWithArgs execution is not permitted in child-agent turns.";
    }
    if ("command".startsWith(parameterName)) {
      const commandParts = attachedValue === null
        ? args.slice(index + 1)
        : [attachedValue, ...args.slice(index + 1)];
      if (!commandParts[0] || commandParts[0] === "-") {
        return "PowerShell command execution is missing its command operand.";
      }
      const command = commandParts.join(" ");
      const tokens = shellTokens(command, true).map((value) => String(value).toLowerCase());
      const launchesScript = shellSegments(command, true).some((segment) => {
        const programIndex = firstProgramIndex(segment);
        if (programIndex < 0 || programIndex >= segment.length) return false;
        const program = String(segment[programIndex]);
        return (
          WINDOWS_SCRIPT_PROGRAM.test(program) ||
          WINDOWS_SCRIPT_HOST_PROGRAMS.has(powershellProgramName(program))
        );
      });
      if (
        launchesScript ||
        powershellCommandDefinesOrImportsCommand(command) ||
        powershellCommandCreatesProcess(command) ||
        powershellActiveCode(command).includes("::") ||
        tokens.some(
          (value) =>
            value === "&" ||
            value === "." ||
            POWERSHELL_INDIRECT_PROCESS_LAUNCHERS.has(value),
        ) ||
        /\[(?:system\.)?diagnostics\.process\]\s*::\s*start\b/i.test(command) ||
        /\b(?:system\.)?diagnostics\.processstartinfo\b/i.test(command) ||
        /\bwscript\.shell\b/i.test(command)
      ) {
        return "Indirect PowerShell process launch is not permitted in child-agent turns.";
      }
      return evaluateModelPythonCommand(
        command,
        cwd,
        trustedRepoRoot,
        trustedScriptDigests,
        null,
        true,
      );
    }
    if (POWERSHELL_OPTIONS_WITH_VALUES.has(optionName)) {
      if (attachedValue === null && !args[index + 1]) {
        return `PowerShell option ${rawOption} is missing its operand.`;
      }
      index += attachedValue === null ? 2 : 1;
      continue;
    }
    if (POWERSHELL_OPTIONS_WITHOUT_VALUES.has(optionName)) {
      if (attachedValue !== null) {
        return `PowerShell flag ${rawOption} does not accept an operand.`;
      }
      index += 1;
      continue;
    }
    if (["-?", "-help", "--help"].includes(optionName) && args.length === 1) {
      return null;
    }
    return `Unsupported PowerShell execution parameter is not permitted in child-agent turns: ${rawOption}`;
  }
  return "PowerShell stdin execution is not permitted in child-agent turns.";
}

function inspectUvRun(args, cwd, trustedRepoRoot, trustedScriptDigests) {
  let index = 1;
  while (index < args.length) {
    const token = args[index];
    if (token === "--") {
      const nested = args.slice(index + 1);
      return nested.length
        ? inspectSegment(nested, cwd, trustedRepoRoot, trustedScriptDigests)
        : null;
    }
    if (!token.startsWith("-")) {
      return inspectSegment(
        args.slice(index),
        cwd,
        trustedRepoRoot,
        trustedScriptDigests,
      );
    }

    const equalsIndex = token.indexOf("=");
    const option = equalsIndex >= 0 ? token.slice(0, equalsIndex) : token;
    const attachedValue = equalsIndex >= 0 ? token.slice(equalsIndex + 1) : null;
    if (UV_RUN_PYTHON_MODULE_OPTIONS.has(option)) {
      return "Python module execution through uv run is not permitted in child-agent turns.";
    }
    if (UV_RUN_PYTHON_SCRIPT_OPTIONS.has(option)) {
      const script = attachedValue ?? args[index + 1];
      if (!script) return `uv run option ${option} is missing its script operand.`;
      return pythonScriptDecision(
        script,
        cwd,
        trustedRepoRoot,
        trustedScriptDigests,
      );
    }
    if (UV_RUN_OPTIONS_WITH_VALUES.has(option)) {
      if (attachedValue !== null) {
        if (!attachedValue) return `uv run option ${option} is missing its operand.`;
        index += 1;
        continue;
      }
      if (!args[index + 1]) return `uv run option ${option} is missing its operand.`;
      index += 2;
      continue;
    }
    if (
      UV_RUN_OPTIONS_WITHOUT_VALUES.has(option) ||
      (/^-[qv]+$/.test(option) && attachedValue === null)
    ) {
      if (attachedValue !== null) {
        return `uv run flag ${option} does not accept an attached operand.`;
      }
      index += 1;
      continue;
    }
    return `Ambiguous uv run option is not permitted in child-agent turns: ${token}`;
  }
  return null;
}

function inspectSegment(segment, cwd, trustedRepoRoot, trustedScriptDigests) {
  const programIndex = firstProgramIndex(segment);
  if (programIndex < 0) {
    return "Ambiguous command-wrapper options are not permitted in child-agent turns.";
  }
  if (programIndex >= segment.length) return null;
  const rawProgram = String(segment[programIndex]);
  if (/[$`]/.test(rawProgram)) {
    return `Dynamically resolved command positions are not permitted in child-agent turns: ${rawProgram}`;
  }
  const program = programName(rawProgram);
  const args = segment.slice(programIndex + 1);

  if (INDIRECT_PROCESS_LAUNCHERS.has(program)) {
    return `Indirect process launcher is not permitted in child-agent turns: ${rawProgram}`;
  }
  if (program === "find" && args.some((value) => FIND_EXEC_OPTIONS.has(value))) {
    return "find process-execution actions are not permitted in child-agent turns.";
  }

  if (SHELL_PROGRAMS.has(program)) {
    return inspectShell(args, cwd, trustedRepoRoot, trustedScriptDigests);
  }

  if (POWERSHELL_PROGRAMS.has(program)) {
    return inspectPowerShell(args, cwd, trustedRepoRoot, trustedScriptDigests);
  }

  if (PYTHON_PROGRAM.test(program)) {
    // Isolated mode ignores cwd and PYTHON* import overrides, so this exact
    // standard-library JSON formatter cannot resolve model-authored modules.
    if (args.length === 1 && args[0] === "--version") return null;
    if (args[0] === "-I" && args[1] === "-m" && args[2] === "json.tool") return null;
    if (args[0] && !args[0].startsWith("-")) {
      return pythonScriptDecision(
        args[0],
        cwd,
        trustedRepoRoot,
        trustedScriptDigests,
      );
    }
    return "Inline, module, or stdin Python execution is not permitted in child-agent turns.";
  }

  if (program === "uv" && args[0] === "run") {
    return inspectUvRun(args, cwd, trustedRepoRoot, trustedScriptDigests);
  }

  const directPath = resolveCommandPath(segment[programIndex], cwd);
  if (directPath) {
    if (WINDOWS_SCRIPT_PROGRAM.test(program)) {
      return shellScriptDecision(
        segment[programIndex],
        cwd,
        trustedRepoRoot,
        trustedScriptDigests,
      );
    }
    if (program.endsWith(".py")) {
      return pythonScriptDecision(
        segment[programIndex],
        cwd,
        trustedRepoRoot,
        trustedScriptDigests,
      );
    }
    try {
      const firstLine = fs.readFileSync(directPath, "utf8").split(/\r?\n/, 1)[0];
      if (/^#!.*\b(?:python|pypy)(?:\d+(?:\.\d+)*)?\b/i.test(firstLine)) {
        return pythonScriptDecision(
          segment[programIndex],
          cwd,
          trustedRepoRoot,
          trustedScriptDigests,
        );
      }
      if (/^#!.*\b(?:bash|dash|sh|zsh)\b/i.test(firstLine)) {
        return shellScriptDecision(
          segment[programIndex],
          cwd,
          trustedRepoRoot,
          trustedScriptDigests,
        );
      }
    } catch {
      return `Direct command path is not a provenance-verified, unchanged executable: ${segment[programIndex]}`;
    }
  }
  return null;
}

export function evaluateModelPythonCommand(
  command,
  cwd,
  trustedRepoRoot,
  trustedScriptDigests = null,
  toolName = null,
  preserveBackslashes = false,
) {
  if (typeof command !== "string" || !command.trim()) return null;
  // The declarative file-edit tool reaches hooks as a raw patch body; Bash may
  // carry the same payload only through one literal quoted apply_patch heredoc.
  // Neither accepted form is shell code, so payload JSON, parentheses,
  // backticks, and dollar signs must not be inspected as shell syntax.
  if (isLiteralApplyPatchCommand(command, toolName)) return null;
  if (command.includes("$(") || command.includes("`")) {
    return "Shell command substitutions are not permitted in child-agent turns.";
  }
  if (hasUnsupportedShellGrouping(command, preserveBackslashes)) {
    return "Grouped shell commands are not permitted in child-agent turns.";
  }
  for (const segment of shellSegments(command, preserveBackslashes)) {
    const decision = inspectSegment(
      segment,
      cwd,
      trustedRepoRoot,
      trustedScriptDigests,
    );
    if (decision) return decision;
  }
  return null;
}

export function preToolUsePythonPolicy(
  input,
  trustedRepoRoot,
  trustedScriptDigests = null,
) {
  const toolInput = input?.tool_input;
  const command =
    typeof toolInput?.command === "string"
      ? toolInput.command
      : typeof toolInput?.cmd === "string"
        ? toolInput.cmd
        : null;
  const cwd = typeof input?.cwd === "string" ? input.cwd : process.cwd();
  const reason = evaluateModelPythonCommand(
    command,
    cwd,
    trustedRepoRoot,
    trustedScriptDigests,
    input?.tool_name,
  );
  if (!reason) return {};
  return {
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason:
        `${reason} Use trusted checked-in workflow tools or write declarative JSON/YAML artifacts instead.`,
    },
  };
}

async function runPythonPolicyHook(
  trustedRepoRoot,
  serializedTrustedScriptDigests,
) {
  if (typeof trustedRepoRoot !== "string" || !trustedRepoRoot.trim()) {
    throw new Error(
      "The Python policy hook requires a non-empty trusted repository root.",
    );
  }
  let text = "";
  process.stdin.setEncoding("utf8");
  for await (const chunk of process.stdin) text += chunk;
  const input = JSON.parse(text);
  const trustedScriptDigests =
    serializedTrustedScriptDigests !== null
      ? JSON.parse(serializedTrustedScriptDigests)
      : null;
  process.stdout.write(
    JSON.stringify(
      preToolUsePythonPolicy(
        input,
        trustedRepoRoot,
        trustedScriptDigests,
      ),
    ) + "\n",
  );
}

async function runDenyToolHook() {
  for await (const _chunk of process.stdin) {
    // Drain the hook payload before returning a deterministic denial.
  }
  process.stdout.write(
    JSON.stringify({
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: "deny",
        permissionDecisionReason: "This independent semantic review is tool-free.",
      },
    }) + "\n",
  );
}

async function main() {
  if (process.argv[2] === "--deny-tool-hook") {
    await runDenyToolHook();
    return;
  }
  if (process.argv[2] === "--python-policy-hook") {
    await runPythonPolicyHook(
      process.argv[3],
      process.argv[4] === undefined ? null : process.argv[4],
    );
    return;
  }
  if (process.argv[2] === "--server") {
    const sessionRequestPath = process.argv[3];
    if (!sessionRequestPath) {
      throw new Error("Usage: codex_sdk_bridge.mjs --server <session-request.json>");
    }
    await runServer(sessionRequestPath);
    return;
  }
  if (process.argv[2] === "--sandbox-smoke") {
    const workspace = process.argv[3];
    if (!workspace) {
      throw new Error("Usage: codex_sdk_bridge.mjs --sandbox-smoke <workspace>");
    }
    await runSandboxSmoke(workspace);
    return;
  }

  const requestPath = process.argv[2];
  if (!requestPath) {
    throw new Error("Usage: codex_sdk_bridge.mjs <request.json>");
  }

  const request = readJsonRequest(requestPath);
  assertCodexHostSupported();
  const codexExecutable = resolveCodexExecutable();
  const codexLauncher = prepareCodexLauncher(
    process.env,
    codexExecutable,
    path.dirname(path.resolve(requestPath)),
    request,
  );
  const restoreWorkingDirectory = activateCodexLauncher(codexLauncher);
  try {
    const { Codex } = await loadCodexSdk();
    const finalResponse = await runTurnWithRetry(
      () => startThread(Codex, request, codexLauncher),
      request,
    );

    if (finalResponse) {
      process.stdout.write(finalResponse);
      if (!finalResponse.endsWith("\n")) {
        process.stdout.write("\n");
      }
    }
  } finally {
    try {
      restoreWorkingDirectory();
    } finally {
      codexLauncher.cleanup();
    }
  }
}

// Keep this separate from a direct `codex exec` check: the SDK's generated
// codexPathOverride has its own executable and temporary-directory boundary.
async function runSandboxSmoke(workspace) {
  const runDir = path.resolve(workspace);
  const marker = path.join(runDir, ".content-workflow-codex-sandbox-smoke");
  const request = {
    repo_root: runDir,
    run_dir: runDir,
    trusted_repo_root: runDir,
    sandbox_writable_roots: [runDir],
    child_final_path: path.join(runDir, "bridge-final.txt"),
    items_path: path.join(runDir, "bridge-items.json"),
    codex_sandbox_mode: DEFAULT_CODEX_SANDBOX_MODE,
    host_executables: sandboxSmokeHostExecutables(),
    prompt: `Run exactly this ${process.platform === "win32" ? "PowerShell" : "shell"} command using the command tool: ${sandboxSmokeWriteCommand(marker, "bridge-ok")}. Then reply exactly OK.`,
  };
  assertCodexHostSupported();
  const codexExecutable = resolveCodexExecutable();
  const launcher = prepareCodexLauncher(
    process.env,
    codexExecutable,
    runDir,
    request,
  );
  const restoreWorkingDirectory = activateCodexLauncher(launcher);
  try {
    const { Codex } = await loadCodexSdk();
    await runTurnWithRetry(() => startThread(Codex, request, launcher), request, 1);
    if (!readExactSandboxSmokeMarker(marker, "bridge-ok")) {
      throw new Error("Codex SDK bridge sandbox smoke did not create its workspace marker");
    }
    process.stdout.write("Codex SDK bridge workspace-write sandbox smoke passed.\n");
  } finally {
    try {
      cleanupSandboxSmokeArtifacts([
        marker,
        request.child_final_path,
        request.items_path,
      ]);
    } finally {
      try {
        restoreWorkingDirectory();
      } finally {
        launcher.cleanup();
      }
    }
  }
}

async function runServer(sessionRequestPath) {
  const sessionRequest = readJsonRequest(sessionRequestPath);
  assertCodexHostSupported();
  const codexExecutable = resolveCodexExecutable();
  const codexLauncher = prepareCodexLauncher(
    process.env,
    codexExecutable,
    path.dirname(path.resolve(sessionRequestPath)),
    sessionRequest,
  );
  const restoreWorkingDirectory = activateCodexLauncher(codexLauncher);
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
    try {
      restoreWorkingDirectory();
    } finally {
      codexLauncher.cleanup();
    }
  }
}

function activateCodexLauncher(launcher) {
  if (!launcher.workingDirectory) {
    return () => {};
  }
  const previous = process.cwd();
  process.chdir(launcher.workingDirectory);
  return () => process.chdir(previous);
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
  if (!launcher.codexConfig) {
    throw new Error("Codex thread requires launcher-owned frozen config");
  }
  const config = launcher.codexConfig;
  const apiKey = resolveCodexApiKey(request.codex_api_key_env, launcher.env);
  const codex = new Codex({
    env: launcher.env,
    config,
    codexPathOverride: launcher.codexPathOverride,
    ...(apiKey === null ? {} : { apiKey }),
  });
  return codex.startThread(buildThreadOptions(request));
}

export function resolveCodexApiKey(environmentName, sourceEnv = process.env) {
  if (environmentName === undefined || environmentName === null || environmentName === "") {
    return null;
  }
  if (
    typeof environmentName !== "string" ||
    !/^[A-Za-z_][A-Za-z0-9_]*$/.test(environmentName)
  ) {
    throw new Error("Invalid codex_api_key_env environment-variable name");
  }
  const value = sourceEnv[environmentName];
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(
      `Codex provider API key environment variable is missing: ${environmentName}`,
    );
  }
  return value;
}

export function prepareCodexLauncher(
  sourceEnv = process.env,
  codexExecutable = resolveCodexExecutable(),
  launcherParentDir = os.tmpdir(),
  request = null,
) {
  if (!request || typeof request !== "object" || Array.isArray(request)) {
    throw new Error("Codex launcher requires one frozen request object");
  }
  const launcherDir = fs.mkdtempSync(
    path.join(launcherParentDir, ".content-workflow-codex-launcher-"),
  );
  const windowsLauncher = process.platform === "win32";
  // On POSIX the SDK can execute the ESM wrapper directly through its shebang.
  // Windows CreateProcess cannot execute .mjs/.cmd launchers, so the SDK starts
  // node.exe and resolves its literal first argument ("exec") in this private
  // directory to the extensionless CommonJS wrapper below.
  const launcherPath = path.join(launcherDir, windowsLauncher ? "exec" : "codex.mjs");
  let launcherEnv;
  let launchConfiguration;
  try {
    fs.chmodSync(launcherDir, 0o700);
    launchConfiguration = buildCodexLaunchConfiguration(request, launcherDir);
    fs.writeFileSync(
      launcherPath,
      windowsLauncher
        ? buildWindowsCodexLauncher(
            codexExecutable,
            launchConfiguration.launcherConfigOverrides,
          )
        : buildCodexLauncher(
            codexExecutable,
            launchConfiguration.launcherConfigOverrides,
          ),
      {
        flag: "wx",
        mode: 0o700,
      },
    );
    launcherEnv = buildLauncherEnvironment(
      sourceEnv,
      launcherDir,
      request.host_executables,
    );
  } catch (error) {
    fs.rmSync(launcherDir, { recursive: true, force: true });
    throw error;
  }
  let cleaned = false;
  return {
    codexPathOverride: windowsLauncher ? process.execPath : launcherPath,
    workingDirectory: windowsLauncher ? launcherDir : null,
    codexConfig: launchConfiguration.config,
    env: launcherEnv,
    cleanup() {
      if (!cleaned) {
        fs.rmSync(launcherDir, { recursive: true, force: true });
        cleaned = true;
      }
    },
  };
}

export function assertCodexHostSupported(platformName = process.platform) {
  if (platformName === "win32") {
    throw new Error(
      "The Codex runner is not supported on native Windows in release 0.6. " +
        "Run the Content Agent workflow inside WSL2 or on native Linux.",
    );
  }
}

function buildLauncherEnvironment(sourceEnv, launcherDir, hostExecutables) {
  const transientDir = path.join(launcherDir, "tmp");
  fs.mkdirSync(transientDir, { mode: 0o700 });
  const launcherEnv = {
    ...sourceEnv,
    // Nested Codex turns can run inside a parent workspace-write sandbox where
    // the process-wide temp directory is intentionally read-only. Keep all
    // transient app-server and PATH-alias work inside the request-local,
    // private launcher directory instead of broadening the parent sandbox.
    // Keep the shadow Codex home beside, rather than inside, the effective
    // temp directory. On Linux Codex deliberately refuses to materialize its
    // codex-linux-sandbox arg0 alias below std::env::temp_dir(); placing
    // CODEX_HOME below TMPDIR therefore removes the helper used by sandboxed
    // shell turns. All disposable SDK scratch still remains in this child.
    TMPDIR: transientDir,
    TMP: transientDir,
    TEMP: transientDir,
  };
  // Never let user rules or plugins participate in unattended nested turns.
  // The private home carries only authentication and wrapper-authored rules.
  const sourceCodexHome = sourceEnv.CODEX_HOME ||
    path.join(sourceEnv.HOME || os.homedir(), ".codex");
  const shadowCodexHome = path.join(launcherDir, "codex-home");
  fs.mkdirSync(shadowCodexHome, { mode: 0o700 });
  const privateStateNames = ["auth.json", ".credentials.json"];
  if (process.platform === "win32") {
    // A nested Windows sandbox may not refresh the mandatory workspace policy
    // before the child starts, so seed Codex's own validated cache.
    privateStateNames.push("cloud-config-bundle-cache.json");
  }
  for (const stateName of privateStateNames) {
    copyRegularCodexStateFile(
      path.join(sourceCodexHome, stateName),
      path.join(shadowCodexHome, stateName),
    );
  }
  writeHostExecutablePolicy(shadowCodexHome, hostExecutables, sourceEnv);
  launcherEnv.CODEX_HOME = shadowCodexHome;
  writeWindowsCodexCaBundle(shadowCodexHome, launcherEnv);
  return launcherEnv;
}

function writeWindowsCodexCaBundle(codexHome, launcherEnv) {
  if (
    process.platform !== "win32" ||
    (typeof launcherEnv.CODEX_CA_CERTIFICATE === "string" &&
      launcherEnv.CODEX_CA_CERTIFICATE.length > 0)
  ) {
    return;
  }
  // The low-privilege Windows sandbox account can enumerate the system trust
  // store while Rust's native certificate loader still rejects one store
  // entry and fails to build an otherwise public chain. Node exposes the same
  // machine roots as PEM, so give the nested Codex client an explicit bundle.
  const certificateGroups =
    typeof tls.getCACertificates === "function"
      ? [tls.getCACertificates("default"), tls.getCACertificates("system")]
      : [tls.rootCertificates];
  const certificates = [...new Set(certificateGroups.flat())].filter(
    (certificate) =>
      typeof certificate === "string" &&
      certificate.startsWith("-----BEGIN CERTIFICATE-----") &&
      certificate.trimEnd().endsWith("-----END CERTIFICATE-----"),
  );
  if (certificates.length === 0) {
    throw new Error("Windows did not expose any trusted CA certificates");
  }
  const bundlePath = path.join(codexHome, "windows-ca-bundle.pem");
  fs.writeFileSync(bundlePath, `${certificates.join("\n")}\n`, {
    flag: "wx",
    mode: 0o600,
  });
  launcherEnv.CODEX_CA_CERTIFICATE = bundlePath;
}

function writeHostExecutablePolicy(codexHome, rawEntries, sourceEnv) {
  if (!Array.isArray(rawEntries) || rawEntries.length === 0) {
    return;
  }
  const pathEntries = String(sourceEnv.PATH ?? "")
    .split(path.delimiter)
    .filter((entry) => entry.length > 0)
    .map((entry) => path.resolve(entry));
  const rules = [];
  const seenNames = new Set();
  for (const entry of rawEntries) {
    const name = entry?.name;
    const candidatePaths = entry?.paths;
    const allowedPrefixes = entry?.allowed_prefixes ?? [[name]];
    if (
      typeof name !== "string" ||
      !/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(name) ||
      seenNames.has(name) ||
      !Array.isArray(candidatePaths) ||
      candidatePaths.length !== 1 ||
      !Array.isArray(allowedPrefixes) ||
      allowedPrefixes.length === 0 ||
      allowedPrefixes.some(
        (prefix) =>
          !Array.isArray(prefix) ||
          prefix.length === 0 ||
          prefix[0] !== name ||
          prefix.some(
            (token) =>
              typeof token !== "string" ||
              token.length === 0 ||
              /[\r\n]/.test(token),
          ),
      )
    ) {
      throw new Error("Invalid Codex host executable policy entry");
    }
    const candidate = candidatePaths[0];
    if (typeof candidate !== "string" || !path.isAbsolute(candidate)) {
      throw new Error(`Codex host executable path must be absolute: ${name}`);
    }
    const realCandidate = fs.realpathSync(candidate);
    const metadata = fs.lstatSync(realCandidate);
    if (!metadata.isFile() || metadata.isSymbolicLink()) {
      throw new Error(`Codex host executable must be a regular file: ${candidate}`);
    }
    const expectedBasename = process.platform === "win32" ? `${name}.exe` : name;
    if (path.basename(realCandidate).toLowerCase() !== expectedBasename.toLowerCase()) {
      throw new Error(`Codex host executable name does not match its path: ${name}`);
    }
    const candidateParent = path.dirname(realCandidate);
    if (
      !pathEntries.some(
        (entryPath) => path.normalize(entryPath).toLowerCase() ===
          path.normalize(candidateParent).toLowerCase(),
      )
    ) {
      throw new Error(`Codex host executable is not on the controlled PATH: ${name}`);
    }
    seenNames.add(name);
    rules.push(
      `host_executable(name=${JSON.stringify(name)}, paths=[${JSON.stringify(realCandidate)}])`,
    );
    for (const prefix of allowedPrefixes) {
      rules.push(
        `prefix_rule(pattern=${JSON.stringify(prefix)}, decision="allow")`,
      );
    }
  }
  const rulesDir = path.join(codexHome, "rules");
  fs.mkdirSync(rulesDir, { mode: 0o700 });
  const rulesPath = path.join(rulesDir, "content-workflow.rules");
  fs.writeFileSync(rulesPath, `${rules.join("\n")}\n`, {
    flag: "wx",
    mode: 0o600,
  });
}

function copyRegularCodexStateFile(sourcePath, destinationPath) {
  let metadata;
  try {
    metadata = fs.lstatSync(sourcePath);
  } catch (error) {
    if (error.code === "ENOENT") {
      return;
    }
    throw error;
  }
  if (!metadata.isFile() || metadata.isSymbolicLink()) {
    throw new Error(`Codex state path must be a regular file: ${sourcePath}`);
  }
  fs.copyFileSync(sourcePath, destinationPath, fs.constants.COPYFILE_EXCL);
  fs.chmodSync(destinationPath, 0o600);
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
  if (process.platform === "win32") {
    return resolveWindowsCodexExecutable(packageMetadata);
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

function resolveWindowsCodexExecutable(packageMetadata) {
  const platformTarget = {
    x64: {
      packageName: "@openai/codex-win32-x64",
      targetTriple: "x86_64-pc-windows-msvc",
    },
    arm64: {
      packageName: "@openai/codex-win32-arm64",
      targetTriple: "aarch64-pc-windows-msvc",
    },
  }[process.arch];
  if (!platformTarget) {
    throw new Error(`Unsupported Windows Codex architecture: ${process.arch}`);
  }

  let nativePackageJsonPath;
  try {
    nativePackageJsonPath = moduleRequire.resolve(
      `${platformTarget.packageName}/package.json`,
    );
  } catch (error) {
    throw new Error(
      `Unable to resolve the installed ${platformTarget.packageName} package`,
      { cause: error },
    );
  }
  let nativeMetadata;
  try {
    nativeMetadata = JSON.parse(fs.readFileSync(nativePackageJsonPath, "utf8"));
  } catch (error) {
    throw new Error(
      `Unable to read Codex platform package metadata at ${nativePackageJsonPath}`,
      { cause: error },
    );
  }
  const expectedVersionPrefix = `${packageMetadata.version}-win32-`;
  if (
    nativeMetadata.name !== "@openai/codex" ||
    typeof nativeMetadata.version !== "string" ||
    !nativeMetadata.version.startsWith(expectedVersionPrefix)
  ) {
    throw new Error(
      `Installed ${platformTarget.packageName} does not match @openai/codex ` +
      `${packageMetadata.version}`,
    );
  }

  const nativePackageRoot = fs.realpathSync(path.dirname(nativePackageJsonPath));
  const lexicalExecutable = path.join(
    nativePackageRoot,
    "vendor",
    platformTarget.targetTriple,
    "bin",
    "codex.exe",
  );
  let realExecutable;
  try {
    realExecutable = fs.realpathSync(lexicalExecutable);
  } catch (error) {
    throw new Error(`Unable to resolve the Codex executable at ${lexicalExecutable}`, {
      cause: error,
    });
  }
  const relative = path.relative(nativePackageRoot, realExecutable);
  if (
    relative === ".." ||
    relative.startsWith(`..${path.sep}`) ||
    path.isAbsolute(relative) ||
    !fs.statSync(realExecutable).isFile()
  ) {
    throw new Error(
      `Installed Codex executable is not package-local: ${realExecutable}`,
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

function buildCodexLauncher(codexExecutable, launcherConfigOverrides = []) {
  return `#!${process.execPath}\n` +
    `import { spawn } from "node:child_process";\n` +
    `import process from "node:process";\n` +
    `const args = process.argv.slice(2);\n` +
    `if (args[0] !== "exec" || args[1] !== "--experimental-json") {\n` +
    `  process.stderr.write("Codex SDK launcher accepts only exec --experimental-json\\n");\n` +
    `  process.exit(2);\n` +
    `}\n` +
    `const launcherConfigOverrides = ${JSON.stringify(launcherConfigOverrides)};\n` +
    `const forwarded = ["exec", "--ignore-user-config", "--dangerously-bypass-hook-trust", args[1]];\n` +
    `for (const override of launcherConfigOverrides) forwarded.push("--config", override);\n` +
    `forwarded.push(...args.slice(2));\n` +
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

function buildWindowsCodexLauncher(
  codexExecutable,
  launcherConfigOverrides = [],
) {
  const javascriptExecutable = [".js", ".cjs", ".mjs"].includes(
    path.extname(codexExecutable).toLowerCase(),
  );
  const childCommand = javascriptExecutable ? process.execPath : codexExecutable;
  const childArgumentPrefix = javascriptExecutable ? [codexExecutable] : [];
  return `const { spawn } = require("node:child_process");\n` +
    `const process = require("node:process");\n` +
    `const args = process.argv.slice(2);\n` +
    `if (args[0] !== "--experimental-json") {\n` +
    `  process.stderr.write("Codex SDK launcher accepts only exec --experimental-json\\n");\n` +
    `  process.exit(2);\n` +
    `}\n` +
    `const launcherConfigOverrides = ${JSON.stringify(launcherConfigOverrides)};\n` +
    `const forwarded = ["exec", "--ignore-user-config", "--dangerously-bypass-hook-trust", args[0]];\n` +
    `for (const override of launcherConfigOverrides) forwarded.push("--config", override);\n` +
    `forwarded.push(...args.slice(1));\n` +
    `const child = spawn(${JSON.stringify(childCommand)}, ` +
    `[...${JSON.stringify(childArgumentPrefix)}, ...forwarded], {\n` +
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
    `child.once("error", (error) => { throw error; });\n` +
    `child.once("exit", (code, signal) => {\n` +
    `  if (signal) {\n` +
    `    for (const [name, handler] of signalHandlers) {\n` +
    `      process.removeListener(name, handler);\n` +
    `    }\n` +
    `    process.kill(process.pid, signal);\n` +
    `  } else {\n` +
    `    process.exit(code ?? 1);\n` +
    `  }\n` +
    `});\n`;
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

// An upstream stream drop is transport flakiness, not a workflow verdict.
// Losing a multi-hour run to one is pure waste, so retry on a fresh thread:
// the run directory persists, so the next turn resumes from its own artifacts.
export const TRANSIENT_TURN_ERROR =
  /stream disconnected|response\.failed|ECONNRESET|ETIMEDOUT|EPIPE|socket hang up|premature close|model is at capacity|\boverloaded\b|\b(?:429|500|502|503|504)\b/i;

export async function runTurnWithRetry(
  makeThread,
  request,
  attempts = 3,
  sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
) {
  for (let attempt = 1; ; attempt += 1) {
    try {
      return await runTurn(makeThread(), request);
    } catch (error) {
      const message = String((error && error.message) || error);
      if (attempt >= attempts || !TRANSIENT_TURN_ERROR.test(message)) {
        throw error;
      }
      process.stderr.write(
        `codex_sdk_bridge: transient turn failure ` +
          `(attempt ${attempt}/${attempts}), retrying on a fresh thread: ` +
          `${message}\n`,
      );
      await sleep(5000 * attempt);
    }
  }
}

export async function runTurn(thread, request) {
  const input = buildInput(request);
  const turnOptions = buildTurnOptions(request);
  const artifacts = prepareRunArtifacts(request);
  let observableArtifact = null;
  try {
    observableArtifact = request.observable_events_path
      ? prepareRunArtifact(request, request.observable_events_path)
      : null;
    const turn =
      typeof thread.runStreamed === "function"
        ? await collectStreamedTurn(
            thread,
            input,
            turnOptions,
            observableArtifact,
          )
        : Object.keys(turnOptions).length > 0
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
    if (observableArtifact) {
      fs.closeSync(observableArtifact.fd);
    }
  }
}

async function collectStreamedTurn(thread, input, turnOptions, observableArtifact) {
  const streamed =
    Object.keys(turnOptions).length > 0
      ? await thread.runStreamed(input, turnOptions)
      : await thread.runStreamed(input);
  const items = [];
  let finalResponse = "";
  let usage = null;
  let terminalType = null;
  if (!streamed?.events?.[Symbol.asyncIterator]) {
    throw new Error("Codex SDK streamed turn omitted its async event iterator");
  }
  for await (const event of streamed.events) {
    if (!event || typeof event !== "object" || typeof event.type !== "string") {
      throw new Error("Codex SDK stream emitted a malformed event without a type");
    }
    if (!SUPPORTED_STREAM_EVENT_TYPES.has(event.type)) {
      throw new Error(`Codex SDK stream emitted unsupported event type: ${event.type}`);
    }
    if (terminalType !== null) {
      throw new Error(
        `Codex SDK stream emitted ${event.type} after terminal event ${terminalType}`,
      );
    }
    const insight = observableInsightFromEvent(event);
    if (insight && observableArtifact) {
      appendPreparedRunArtifact(
        observableArtifact,
        JSON.stringify(sanitizeObservableRecord(insight)) + "\n",
      );
    }
    if (event?.type === "item.completed" && event.item) {
      items.push(event.item);
      if (event.item.type === "agent_message") {
        finalResponse = String(event.item.text ?? "");
      }
    } else if (event?.type === "turn.completed") {
      terminalType = event.type;
      usage = event.usage ?? null;
    } else if (event?.type === "turn.failed") {
      terminalType = event.type;
      throw new Error(safeProviderError(event.error?.message, "Codex turn failed"));
    } else if (event?.type === "turn.cancelled") {
      terminalType = event.type;
      throw new Error("Codex turn cancelled by request");
    } else if (event?.type === "error") {
      throw new Error(safeProviderError(event.message, "Codex event stream failed"));
    }
  }
  if (terminalType !== "turn.completed") {
    throw new Error(
      "stream disconnected before turn.completed from the Codex Responses protocol",
    );
  }
  return { items, finalResponse, usage };
}

function safeProviderError(value, fallback) {
  return boundedObservableText(value ?? fallback, 2000);
}

export function observableInsightFromEvent(event) {
  if (["thread.started", "turn.started"].includes(event?.type)) {
    const scope = event.type === "thread.started" ? "thread" : "turn";
    return {
      schema_version: "content-agents.observable-insight.v1",
      id: `codex-${scope}:started`,
      time: new Date().toISOString(),
      phase: "reasoning",
      source: "codex_sdk",
      status: "running",
      kind: "status",
      title: `Agent ${scope} started`,
      summary: `The agent ${scope} entered the streamed Responses lifecycle.`,
    };
  }
  if (["turn.completed", "turn.failed", "turn.cancelled", "error"].includes(event?.type)) {
    const failed = ["turn.failed", "error"].includes(event.type);
    const cancelled = event.type === "turn.cancelled";
    return {
      schema_version: "content-agents.observable-insight.v1",
      id: `codex-turn:${event.type}`,
      time: new Date().toISOString(),
      phase: "reasoning",
      source: "codex_sdk",
      status: failed ? "error" : cancelled ? "warning" : "success",
      kind: failed ? "error" : cancelled ? "notice" : "status",
      title: failed
        ? "Agent turn failed"
        : cancelled
          ? "Agent turn cancelled"
          : "Agent turn completed",
      summary: failed
        ? safeProviderError(event.error?.message ?? event.message, "Provider error")
        : cancelled
          ? "The agent turn was cancelled before completion."
          : "The agent turn reached its terminal completion event.",
    };
  }
  const item = event?.item;
  if (!item || !["item.started", "item.completed"].includes(event.type)) {
    return null;
  }
  // Reasoning items are intentionally private. The viewer exposes only public
  // agent messages and observable actions/results.
  if (item.type === "reasoning") {
    return null;
  }
  const lifecycle = event.type === "item.started" ? "started" : "completed";
  const base = {
    schema_version: "content-agents.observable-insight.v1",
    id: `${item.id ?? "item"}:${lifecycle}`,
    time: new Date().toISOString(),
    phase: "reasoning",
    source: "codex_sdk",
    status:
      lifecycle === "started"
        ? "running"
        : ["failed", "declined"].includes(item.status)
          ? "error"
          : "success",
  };
  if (item.type === "agent_message" && lifecycle === "completed") {
    return {
      ...base,
      kind: "commentary",
      title: "Agent update",
      summary: boundedObservableText(item.text),
    };
  }
  if (item.type === "command_execution") {
    return {
      ...base,
      kind: "action",
      title: lifecycle === "started" ? "Agent action started" : "Agent action finished",
      summary: summarizeCommand(item.command),
      next_action:
        lifecycle === "started"
          ? "Wait for the observable command result."
          : `Command status: ${item.status ?? "completed"}.`,
    };
  }
  if (item.type === "mcp_tool_call") {
    return {
      ...base,
      kind: "action",
      title: "Agent tool call",
      summary: `${item.server ?? "MCP"} · ${item.tool ?? "tool"}`,
    };
  }
  if (item.type === "file_change" && lifecycle === "completed") {
    const changes = Array.isArray(item.changes) ? item.changes : [];
    return {
      ...base,
      kind: "action",
      title: "Agent artifact update",
      summary: changes.length
        ? changes.map((change) => `${change.kind}: ${change.path}`).join(", ")
        : "Updated workflow artifacts.",
    };
  }
  if (item.type === "todo_list") {
    const todos = Array.isArray(item.items) ? item.items : [];
    return {
      ...base,
      kind: "plan",
      title: "Agent plan",
      summary: todos
        .map((todo) => `${todo.completed ? "Done" : "Next"}: ${todo.text}`)
        .join(" · "),
    };
  }
  if (item.type === "web_search" && lifecycle === "completed") {
    return {
      ...base,
      kind: "action",
      title: "Agent search",
      summary: boundedObservableText(item.query),
    };
  }
  if (item.type === "error" && lifecycle === "completed") {
    const summary = boundedObservableText(item.message);
    if (
      summary ===
      "`--dangerously-bypass-hook-trust` is enabled. Enabled hooks may run without review for this invocation."
    ) {
      return {
        ...base,
        kind: "notice",
        title: "Agent audit notice",
        summary,
        status: "warning",
      };
    }
    return {
      ...base,
      kind: "error",
      title: "Agent error",
      summary,
      status: "error",
    };
  }
  return null;
}

function summarizeCommand(command) {
  const normalized = String(command ?? "").replace(/\s+/g, " ").trim();
  return boundedObservableText(normalized || "Executed a shell command.", 500);
}

function boundedText(value, limit = 1200) {
  const text = String(value ?? "").trim();
  return text.length <= limit ? text : `${text.slice(0, limit)}…`;
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
    );
  return boundedText(redacted, limit);
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

export function appendPreparedRunArtifact(artifact, content) {
  assertPreparedRunArtifact(artifact);
  fs.writeSync(artifact.fd, content, null, "utf8");
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

export function buildCodexConfig(request, launcherWritableRoot = null) {
  return buildCodexLaunchConfiguration(request, launcherWritableRoot).config;
}

export function buildCodexLaunchConfiguration(
  request,
  launcherWritableRoot = null,
) {
  const codexConfig = { ...(request.codex_config ?? {}) };
  dropSecurityCriticalCodexConfigKeys(codexConfig);
  const sandboxMode = resolveCodexSandboxMode(
    request.codex_sandbox_mode,
    request.allow_unsafe_host_child === true,
  );
  const credentialsStore = resolveCodexAuthCredentialsStore(
    request.cli_auth_credentials_store,
  );
  // `codex_config` is a trusted local escape hatch forwarded to the Codex SDK
  // for provider/auth customization. Keep unattended execution controls fixed
  // after the spread so request config cannot relax the wrapper sandbox policy.
  const policyHook = request.tools_disabled
    ? "--deny-tool-hook"
    : "--python-policy-hook";
  const config = {
    ...codexConfig,
    approval_policy: "never",
    features: {
      plugins: false,
      hooks: true,
      ...(request.tools_disabled
        ? {
            apps: false,
            browser_use: false,
            code_mode_host: false,
            computer_use: false,
            goals: false,
            image_generation: false,
            multi_agent: false,
            shell_tool: false,
            unified_exec: false,
          }
        : {}),
    },
    hooks: {
      PreToolUse: [
        {
          hooks: [
            {
              type: "command",
              command: [
                shellQuote(process.execPath),
                shellQuote(BRIDGE_PATH),
                policyHook,
                ...(request.tools_disabled
                  ? []
                  : [
                      shellQuote(request.trusted_repo_root ?? ""),
                      shellQuote(
                        JSON.stringify(request.trusted_script_digests ?? {}),
                      ),
                    ]),
              ].join(" "),
              timeout: 10,
            },
          ],
        },
      ],
    },
  };
  if (credentialsStore !== null) {
    config.cli_auth_credentials_store = credentialsStore;
  }
  let permissionProfile = null;
  if (sandboxMode === DEFAULT_CODEX_SANDBOX_MODE) {
    const launchNetworkPolicy = request.child_launch?.network_policy;
    const allowedHosts = Array.isArray(launchNetworkPolicy?.allowed_hosts)
      ? launchNetworkPolicy.allowed_hosts.filter(
          (host) => typeof host === "string" && host.length > 0,
        )
      : [];
    const reasoningTransportOnly =
      launchNetworkPolicy?.mode === "reasoning_transport_only";
    if (
      reasoningTransportOnly &&
      (launchNetworkPolicy.tool_network_access === true ||
        allowedHosts.some((host) => !REASONING_ADAPTER_HOSTS.has(host)))
    ) {
      throw new Error(
        "Reasoning-only child launch network policy can allow only loopback adapters",
      );
    }
    if (
      !reasoningTransportOnly &&
      launchNetworkPolicy !== undefined &&
      launchNetworkPolicy.tool_network_access !== true &&
      allowedHosts.length > 0
    ) {
      throw new Error(
        "Child launch network policy cannot allow domains when tool network is disabled",
      );
    }
    const scopedNetworkPolicy = allowedHosts.length > 0;
    const networkAccess =
      launchNetworkPolicy === undefined
        ? true
        : launchNetworkPolicy.tool_network_access === true ||
          (reasoningTransportOnly && scopedNetworkPolicy);
    // Use one launcher-owned permission profile so the private launcher TMPDIR
    // and exact loopback routes are part of the same sandbox authority.  The
    // read-only base keeps general /tmp non-writable; the current workspace,
    // wrapper-approved roots, and this launcher's private directory are the
    // only write grants. Do not put this profile into the SDK config object:
    // SDK 0.147.0 recursively flattens it and destroys the quoted
    // `:workspace_roots` FilesystemPermissionToml key. The private launcher
    // injects the one serialized inline table before the remaining SDK args.
    config.default_permissions = CONTENT_WORKFLOW_CODEX_PERMISSION_PROFILE;
    permissionProfile = {
      extends: ":read-only",
      filesystem: {
        ":workspace_roots": "write",
      },
      network: {
        enabled: networkAccess,
      },
    };
    if (scopedNetworkPolicy) {
      config.features.network_proxy = true;
      config.network_proxy = {
        enabled: true,
        enable_socks5: false,
        enable_socks5_udp: false,
        allow_upstream_proxy: false,
        // The exact loopback literal is allowlisted below. Keep the broader
        // local/private-network bypass disabled.
        allow_local_binding: false,
      };
      // The SDK recursively flattens config objects into dotted TOML keys.
      // Quote host keys explicitly so an IPv4 or IPv6 literal remains one
      // `network_proxy.domains` key instead of being split at each dot.
      for (const host of allowedHosts) {
        const normalizedHost =
          host.startsWith("[") && host.endsWith("]")
            ? host.slice(1, -1)
            : host;
        config[`network_proxy.domains.${JSON.stringify(normalizedHost)}`] =
          "allow";
        permissionProfile.network.domains ??= {};
        permissionProfile.network.domains[normalizedHost] = "allow";
      }
    }
    // The run directory can live outside the agent working directory (the
    // wrapper launches the child from <repo-root>/agentic while run artifacts
    // default under the caller's CWD); grant it explicitly so the child can
    // write the required artifacts. Wrapper-owned security config cannot add
    // roots because request codex_config overrides are dropped above.
    const writableRoots = sanitizeWritableRoots(request.sandbox_writable_roots);
    if (
      typeof launcherWritableRoot === "string" &&
      path.isAbsolute(launcherWritableRoot)
    ) {
      writableRoots.push(launcherWritableRoot);
    }
    for (const writableRoot of new Set(writableRoots)) {
      permissionProfile.workspace_roots ??= {};
      permissionProfile.workspace_roots[writableRoot] = true;
    }
  } else {
    config.sandbox_mode = sandboxMode;
  }
  if (request.codex_responses_url) {
    const derivedBaseUrl = validateCodexResponsesUrl(request.codex_responses_url);
    if (
      request.codex_base_url &&
      validateCodexBaseUrl(request.codex_base_url) !== derivedBaseUrl
    ) {
      throw new Error(
        "codex_base_url does not match the base derived from codex_responses_url",
      );
    }
    configureResponsesModelProvider(
      config,
      codexConfig,
      derivedBaseUrl,
      request.codex_api_key_env,
    );
  } else if (request.codex_base_url) {
    config.openai_base_url = validateCodexBaseUrl(request.codex_base_url);
  }
  return {
    config,
    launcherConfigOverrides:
      permissionProfile === null
        ? []
        : [serializeCodexPermissionProfile(permissionProfile)],
  };
}

function serializeCodexPermissionProfile(permissionProfile) {
  return (
    `permissions.${CONTENT_WORKFLOW_CODEX_PERMISSION_PROFILE}=` +
    renderInlineTomlTable(permissionProfile)
  );
}

function renderInlineTomlTable(table) {
  const fields = Object.entries(table).map(([key, value]) => {
    const renderedKey = /^[A-Za-z0-9_-]+$/.test(key) ? key : JSON.stringify(key);
    const renderedValue =
      value && typeof value === "object" && !Array.isArray(value)
        ? renderInlineTomlTable(value)
        : typeof value === "string"
          ? JSON.stringify(value)
          : typeof value === "boolean"
            ? String(value)
            : null;
    if (renderedValue === null) {
      throw new Error(`Unsupported launcher permission value at ${key}`);
    }
    return `${renderedKey} = ${renderedValue}`;
  });
  return `{ ${fields.join(", ")} }`;
}

export function sanitizeWritableRoots(value) {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter(
    (root) => typeof root === "string" && root.length > 0 && path.isAbsolute(root),
  );
}

function resolveCodexAuthCredentialsStore(value) {
  if (!SUPPORTED_CODEX_AUTH_CREDENTIALS_STORES.has(value)) {
    return null;
  }
  return value;
}

const LOOPBACK_HOSTNAMES = new Set(["localhost", "127.0.0.1", "[::1]", "::1"]);

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
  // The API key travels to this endpoint on every turn, including retries.
  // Over plaintext http that key crosses the network in the clear, so a
  // remote endpoint must be https. Loopback stays allowed: a local proxy
  // never leaves the host, and requiring a certificate for it would push
  // people toward disabling verification instead.
  if (parsed.protocol === "http:" && !LOOPBACK_HOSTNAMES.has(parsed.hostname)) {
    throw new Error(
      `Invalid codex_base_url: ${value} (https is required for non-loopback hosts; ` +
        "the API key would otherwise be sent in cleartext)",
    );
  }
  return parsed.toString().replace(/\/$/, "");
}

export function validateExplicitResponsesProvider(codexConfig, derivedBaseUrl) {
  const providerName = codexConfig.model_provider;
  if (providerName === undefined) {
    return;
  }
  if (typeof providerName !== "string" || !providerName.trim()) {
    throw new Error("Codex model_provider must be a non-empty string");
  }
  const providers = codexConfig.model_providers;
  const provider =
    providers && typeof providers === "object" && !Array.isArray(providers)
      ? providers[providerName]
      : null;
  if (!provider || typeof provider !== "object" || Array.isArray(provider)) {
    throw new Error(`Codex model_provider has no matching provider: ${providerName}`);
  }
  if (provider.wire_api !== "responses") {
    throw new Error(
      `Codex model provider ${providerName} must declare wire_api=responses ` +
        "for codex_responses_url",
    );
  }
  if (typeof provider.base_url === "string") {
    if (validateCodexBaseUrl(provider.base_url) !== derivedBaseUrl) {
      throw new Error(
        `Codex model provider ${providerName} base_url does not match ` +
          "codex_responses_url",
      );
    }
  }
}

export function configureResponsesModelProvider(
  config,
  codexConfig,
  baseUrl,
  apiKeyEnvironmentName = null,
) {
  validateExplicitResponsesProvider(codexConfig, baseUrl);
  if (
    apiKeyEnvironmentName !== null &&
    apiKeyEnvironmentName !== undefined &&
    apiKeyEnvironmentName !== "" &&
    (typeof apiKeyEnvironmentName !== "string" ||
      !/^[A-Za-z_][A-Za-z0-9_]*$/.test(apiKeyEnvironmentName))
  ) {
    throw new Error("Invalid codex_api_key_env environment-variable name");
  }
  const providerName =
    codexConfig.model_provider ?? CONTENT_WORKFLOW_CUSTOM_MODEL_PROVIDER;
  const configuredProviders =
    codexConfig.model_providers &&
    typeof codexConfig.model_providers === "object" &&
    !Array.isArray(codexConfig.model_providers)
      ? codexConfig.model_providers
      : {};
  const configuredProvider = configuredProviders[providerName] ?? {};
  config.model_providers = {
    ...configuredProviders,
    [providerName]: {
      name: configuredProvider.name ?? providerName,
      ...configuredProvider,
      base_url: baseUrl,
      wire_api: "responses",
      env_key:
        apiKeyEnvironmentName || configuredProvider.env_key || "OPENAI_API_KEY",
    },
  };
  config.model_provider = providerName;
  // openai_base_url selects the legacy websocket transport. A Responses URL
  // must be represented only by a named responses provider so Codex uses
  // HTTP POST + SSE.
  delete config.openai_base_url;
}

export function validateCodexResponsesUrl(value) {
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error(`Invalid codex_responses_url: ${value}`);
  }
  if (
    !["http:", "https:"].includes(parsed.protocol) ||
    !parsed.host ||
    parsed.username ||
    parsed.password ||
    parsed.search ||
    parsed.hash ||
    !parsed.pathname.replace(/\/$/, "").endsWith("/responses")
  ) {
    throw new Error(`Invalid codex_responses_url: ${value}`);
  }
  if (parsed.protocol === "http:" && !LOOPBACK_HOSTNAMES.has(parsed.hostname)) {
    throw new Error(
      `Invalid codex_responses_url: ${value} (https is required for non-loopback hosts; ` +
        "the API key would otherwise be sent in cleartext)",
    );
  }
  parsed.pathname = parsed.pathname.replace(/\/$/, "").slice(0, -"/responses".length);
  return parsed.toString().replace(/\/$/, "");
}

function resolveCodexSandboxMode(value, allowUnsafeHostChild = false) {
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
  if (
    value === DANGER_FULL_ACCESS_CODEX_SANDBOX_MODE &&
    !allowUnsafeHostChild
  ) {
    throw new Error(
      `Unsupported Codex sandbox mode: ${value}. ` +
        "The trusted launcher did not authorize an unsafe host child.",
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
