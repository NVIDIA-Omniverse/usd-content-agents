# SDF Backend Admission

## Trust Boundary

An SDF backend is implementation code, not model-authored content. Installed
drivers are discovered through the `sdf_tools.backends` entry-point group, but
only an exact backend ID, distribution, entry-point target, implementation
version, license-manifest digest, and native closure-policy digest named in
checked-in policy may load. Agent input may select an admitted ID; it may not
supply a package, module path, entry point, factory, or registration object.

Every driver must declare semantic operations, implementation identity,
immutable read and write artifact-format sets, deterministic priority, and
`execution_mode` equal to `in_process`. The compatibility `supported_formats`
identity field is the union of those directional sets. OpenVDB 13 is the first
admitted driver, not the definition of the public SDF contract.

## License Gate

Backend admission requires a structured component manifest and a verified
native dependency-closure attestation. The SDF delta must not introduce or
bundle an LGPL component. Reject forbidden, unknown, incomplete, or mismatched
introduced/bundled license records; do not infer safety from the backend's
top-level license string.

For a native driver, the attestation must identify a reviewed architecture
record that binds the source lock, build and license gates, repaired wheel, and
every installed native member. Generated candidate evidence does not qualify a
backend and must fail closed until separately reviewed and promoted.

Platform ABI and pre-existing application components must be classified
separately so the delta-scoped claim remains precise. They cannot be relabeled
as introduced, bundled, or newly approved by an SDF manifest. The OpenVDB
driver's application-provided NumPy is in this category; nanobind and robin-map
are compiled/bundled inputs and are not.

## Execution Boundary

The selected driver executes in the importing CPython process. A content agent
calls `sdf_tools`; the admitted driver may load a native Python extension in
that same process. Geometry Repair may already be running inside its generic
resource-limited worker process, but SDF dispatch inside that worker remains
in-process. Do not add a backend-specific subprocess, daemon, socket, RPC, or
file-based transport.

An in-process parser is not admitted for untrusted artifacts merely because it
checks file size or metadata before a full read. A driver may declare
`READ_FIELDS` only after its parser boundary provides the allocation and handle
guarantees required by policy.

## OpenVDB Driver Restrictions

The current OpenVDB driver declares `WRITE_FIELDS` but not `READ_FIELDS`.
OpenVDB's metadata preflight and post-read checks cannot prevent allocations
during native parsing, so they are not an untrusted-input boundary. A session
that requires `READ_FIELDS` must fail capability admission; low-level
trusted-artifact maintenance reads remain outside the content-agent capability
surface.

Promotion also requires capability conformance, resource-limit tests,
deterministic repeats, exact implementation provenance, packaging tests, and
domain validation for each workflow that grants the capability authority.
