# Dependency Localization

Read this reference before opening external USD layers, references, payloads,
textures, or other asset paths for repair or visual evidence.

## Authority

`localize_usd_dependencies` may resolve an authored identifier only from:

- the selected source directory;
- a caller-approved absolute root; or
- one exact authored-identifier remap, optionally scoped to an exact source
  layer and protected by an expected SHA-256.

Do not search by basename, prefix, suffix, glob, or similarity. Do not fetch a
remote URI. Reject parent traversal, symlink escapes, unsafe package members,
hash mismatch, duplicate selectors, and conflicting mappings.

## Portable Bundle

Copy the immutable source and every approved dependency into a deterministic
bundle. Rewrite only the copied layers. Record, for every mapping, the authored
identifier, source layer, resolution method, source path/hash/size, package
path, localized hash/size, package member, and whether the rewrite was applied.

Retain `dependency_localization.json` in the certificate, geometry validation
evidence, and `content_agents_manifest.json`.

## Claim Boundary

- `dependency_complete=true` means every discovered dependency was safely
  copied and rewritten; it still does not prove material quality.
- Any unresolved or unrewritten identifier sets
  `material_fidelity_status=not_claimed_unresolved_dependencies`.
- Geometry diagnosis may continue from a composition-complete source or
  localized bundle, but unresolved material dependencies remain visible.
- Never invent or substitute a texture to make localization pass.
