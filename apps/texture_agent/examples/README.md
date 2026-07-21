# Texture Agent Examples

These examples are small, reproducible entry points for Texture Agent users.
They live with the Texture Agent app because they exercise app-specific CLI and
service fields. Each bucket example includes setup, exact request/config
fields, pass criteria, checked-in reference images, and a command to create a
local visual evidence sheet from the run artifacts.

| Example | Backend | What It Demonstrates |
|---|---|---|
| [Simple Image-Gen Bucket](simple_image_gen_bucket/) | `simple_image_gen` CLI | A lightweight text-to-texture baseline on the public SimReady cleaning bucket, including final render, albedo, and ORM reference outputs. |

Use the simple image-gen example first when you want to verify the core Texture
Agent CLI flow. Public 0.5 staging keeps the Texture Variation API service
backend contract, but does not ship the managed Step1X runtime package or
Step1X evidence workflow.
