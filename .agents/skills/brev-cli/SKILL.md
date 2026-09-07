---
name: brev-cli
description: Manage Brev instances safely from the Brev CLI. Use when a workflow needs to list, create, access, copy to, execute on, port-forward, stop, or delete Brev instances for Content Agents dependency endpoints.
version: "0.1.0"
author: NVIDIA Omniverse
tags:
  - content-agents
  - brev
  - deployment
tools:
  - Shell
  - Filesystem
  - brev
  - ssh
compatibility: Requires an installed and authenticated Brev CLI, SSH access to created instances, and explicit user approval before commands that start spend or delete resources.
metadata:
  author: NVIDIA Omniverse
  tags:
    - content-agents
    - brev
    - deployment
---

# Brev CLI

Use this skill for generic Brev instance lifecycle operations that support
Content Agents deployment skills.

## When to Use

- A deployment skill asks you to list, create, access, port-forward, stop, or
  delete Brev instances.
- The user asks for direct Brev CLI help outside a service-specific deployment
  skill.
- A collection deployment needs a generic Brev lifecycle step before handing
  off to `$fixed-pipeline` and its `deploy-ovrtx-docker`,
  `deploy-qwen-vlm-brev`, `deploy-image-gen-brev`, or
  `deploy-embeddings-brev` reference.

## Limitations

- Do not create paid instances or delete existing instances without explicit
  user approval for the exact target.
- Do not print, log, or copy secrets unless the user has provided a safe secret
  handling path.
- Do not assume instance-to-instance private networking is available. Prefer
  explicit port-forwards or user-provided endpoint URLs.
- Do not invent GPU type names. Use `brev ls --json`, `brev search --json`, or
  a service-specific skill's validated instance type.

## Prerequisites

- Brev CLI installed and authenticated.
- Network access from the local machine to the Brev control plane.
- SSH readiness for commands that use `brev exec`, `brev copy`, or direct
  `ssh`.
- The `$fixed-pipeline` umbrella and selected deployment reference loaded when
  configuring OVRTX, model, image-generation, or embedding containers.

## Instructions

1. Check local CLI state, current resources, and available hardware. Skip
   `brev search --json` only when the selected fixed-pipeline reference already gives an
   exact validated instance type:

```bash
brev --version
brev healthcheck
brev ls --json
brev search --json
```

2. Reuse an existing instance only when its name or purpose matches the
   requested role, it is not deleting, its status and shell access are usable,
   its GPU/type/disk satisfy the service-specific requirement, and no other
   owner or run is actively using it. If no instance matches, run a dry-run
   first; `<gb>` is the Brev CLI disk-size value in GB:

```bash
brev create <name> --dry-run --type <instance-type> --min-disk <gb>
```

3. After the user approves the exact instance name and type, create it:

```bash
brev create <name> --type <instance-type> --min-disk <gb> --timeout 1200
```

4. Wait until the instance is shell-ready before copying files or starting
   containers:

```bash
brev exec <name> "uname -a && nvidia-smi || true"
```

5. Copy only the files needed by the service-specific workflow. Keep local
   `.env` files and secrets out of bulk copies unless the user explicitly asks
   for a credential-copy step:

```bash
brev copy <local-path> <name>:<remote-path>
```

6. Run remote commands through `brev exec` or a named SSH host. Prefer
   the selected fixed-pipeline deployment reference for container startup commands:

```bash
brev exec <name> "cd <remote-path> && <command>"
```

7. Forward endpoints explicitly and keep the forwarding process running while
   local agents use it. Prefer a foreground terminal or tool session. If the
   process must run in the background, save its PID or terminal/session name so
   cleanup can stop the exact port-forward:

```bash
brev port-forward <name> -p <local-port>:<remote-port>
```

8. Clean up port-forward processes, then choose exactly one instance cleanup
   action when validation is complete. Use `brev stop` when delete is delayed,
   delete is not supported, or the user asks to preserve the workspace without
   spend:

```bash
brev stop <name>
```

Use `brev delete` only when the user wants the instance removed:

```bash
brev delete <name>
```

After either cleanup action, confirm state:

```bash
brev ls --json
```

## Command Reference

| Goal | Command |
|---|---|
| Check auth and CLI health | `brev healthcheck` |
| List instances | `brev ls --json` |
| Find available hardware | `brev search --json` |
| Preview spend | `brev create <name> --dry-run --type <type> --min-disk <gb>` |
| Create instance | `brev create <name> --type <type> --min-disk <gb> --timeout 1200` |
| Run remote command | `brev exec <name> "<command>"` |
| Copy files | `brev copy <local-path> <name>:<remote-path>` |
| Forward port | `brev port-forward <name> -p <local-port>:<remote-port>` |
| Stop spend when supported | `brev stop <name>` |
| Delete instance | `brev delete <name>` |

## Output Format

Report the instance name, instance type, current status, commands run, endpoint
or port-forward URL, any port-forward PID/session to stop, and the `brev stop`
or `brev delete` cleanup command. Mention any command that requires user
approval before running it.

## Troubleshooting

- If `brev healthcheck` fails, ask the user to refresh Brev authentication and
  rerun the command.
- If shell readiness fails after creation, poll `brev ls --json` and retry a
  lightweight `brev exec` before starting containers.
- If `brev port-forward` only binds localhost but Docker containers need the
  endpoint, use a carefully scoped SSH forward bound to the host interface and
  restrict access with host firewall rules.
- If delete is delayed, stop the instance when supported and poll
  `brev ls --json` until the instance disappears.
