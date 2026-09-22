# Source-clear initialization for the unscored drawer follow-up

The original pilot starts the payload lower face 8 mm below the visible upper
drawer floor. Its accepted direct-Astra collision proxy starts 4.50 mm below
that surface, so the original 5 mm penetration gate describes the approximation.
Those results and the frozen pilot evaluator remain unchanged.

This follow-up derives one initialization correction from the original CC0
Poly Haven glTF, before running the native authored drawer against it. The
builder reads no authored submission. It clips every original upper-drawer
triangle to a conservative XZ box covering the declared position jitter and
all yaw variations. The maximum surface height is 1.0129997730255127 m.
Placing the payload lower face 10 mm above it gives the new center
`[0, 1.0529997730255127, 0]` m. Clipping all five original meshes against the
entire swept initial payload box finds zero surface intersections.

Only `protocol_id` and `payload.initial_center_m[1]` change. Source arrays,
evaluator and solver bytes, five seeds, controller, gravity, payload dimensions
and mass, cavity queries, retention bounds and every acceptance threshold stay
identical. The original twelve-file freeze is verified before and after the
copy. Eleven non-specification files remain byte-identical.

The builder has five analytic clipping checks. The copied, unchanged evaluator
tests also pass all nine synthetic cases and seven static-parenting cases.
Native synthetic simulation passes the moving hollow drawer and correctly
rejects blocked motion and a filled cavity. These are fixture qualification,
not an acceptance result for the source cabinet or a Content Agents output.

- [Source-clear freeze and input hashes](evaluator/source_clear_v1/source_clear_freeze.json)
- [Sixteen-check qualification](evaluator/source_clear_v1/qualification.json)
- [Source-only builder](prepare_source_clear_fixture.py)

The native authored asset still requires cooked-geometry and initial-contact
preflight, followed by five independent loaded-drawer trials. Source clearance
alone does not establish collision-proxy clearance. A final claim must bind the
exact same authored asset to the native workflow records and task evaluation.

The first build attempt failed while serializing a NumPy Boolean after its
geometry checks. Its partial directory is retained separately. Casting the
analytic check results to Python Boolean values fixed serialization; it changed
no geometric computation or task parameter. The complete freeze was written
to a fresh directory before qualification or authored-asset trials.
