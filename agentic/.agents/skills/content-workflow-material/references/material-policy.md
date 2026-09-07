# Material Assignment Policy

- Use only supplied or configured material libraries.
- Preserve exact material names and paths in artifacts.
- Prefer visual match when exact substance is unavailable, but do not choose a
  proxy that changes the transparency class, metalness class, dominant color,
  or finish in a visibly worse way.
- In clean-slate workflows, every visible/renderable material candidate must be
  assigned a library material. Record ambiguous or unassignable candidates as
  unresolved evidence; they keep the workflow incomplete.
- Freeze clean-slate candidates only after `appearance clear` and a fresh
  visible-mesh query. The clear operation deinstances native instance roots;
  never reinterpret a pre-clear proxy-only result as an empty asset.
- Treat complete material-subset face coverage as a fully bound mesh. Never add
  an undeclared direct parent fallback solely to silence an `unbound` parent
  entry; preserve the subset decisions and use the final binding audit's face
  coverage result.
- Existing material bindings can count as preserved coverage only when the
  workflow explicitly respects existing bindings.
- Group repeated parts when the same material decision applies.
- Record material limitations instead of hiding them in prose.
