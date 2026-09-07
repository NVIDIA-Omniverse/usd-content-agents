# Part Plan Schema

Use a part plan when segmenting product or scene images for CAD reconstruction.

```json
{
  "image": "exploded_generated.png",
  "split_levels": 2,
  "notes": "Optional run notes.",
  "parts": [
    {
      "id": "outer_shell",
      "label": "Main outer C-frame shell",
      "bbox": [0.08, 0.10, 0.48, 0.78],
      "role": "component",
      "material_hint": "cream injection-molded plastic",
      "material_zone": "outer molded shell",
      "function_hint": "Outer support body that locates internal components and visible service opening.",
      "signature_details": ["C-shaped side opening", "broad rounded top housing"],
      "visual_facts": [
        {"kind": "visual_silhouette", "statement": "Tall curved C-frame outer contour"},
        {"kind": "visual_topology", "statement": "One contiguous shell ribbon with a boundary-connected cutout, not separate bars"},
        {"kind": "visual_negative_space", "statement": "Boundary-connected side opening"},
        {"kind": "visual_repetition", "statement": "Nine or ten ribs remain plausible through occlusion", "observed_count": null, "plausible_counts": [9, 10], "count_parameter": "RibCount", "binding": false}
      ],
      "proportion_notes": "Tall C-shaped side silhouette, broad top housing, lower base shorter than top housing.",
      "construction_class": "constant_thickness_planar_profile",
      "profile_view": "right side, normal to shell thickness",
      "interface_notes": "Top openings, side window insert, internal rails, screw bosses, and tray contact surface must align with other parts.",
      "visible_interfaces": ["side window opening", "tray contact surface"],
      "expected_features": ["front service aperture", "top openings", "side window", "internal rails"],
      "crop_confidence": "high",
      "detail_carrier": "geometry",
      "cad_notes": "Large radiused C-frame body with front aperture and side panel seams.",
      "children": [
        {
          "id": "water_window",
          "label": "Side water-window lens",
          "bbox": [0.03, 0.23, 0.16, 0.78],
          "bbox_space": "parent",
          "role": "component",
          "material_hint": "clear plastic",
          "cad_notes": "Thin transparent insert mounted in the shell side opening."
        }
      ]
    }
  ]
}
```

`bbox` may be normalized `[x0, y0, x1, y1]` values from 0 to 1, or absolute pixel coordinates. The split script auto-detects source boxes with values <= 1.5 as normalized for compatibility with older plans. Use `bbox_space: "source_pixels"` or `"pixels"` for tiny absolute boxes that could otherwise look normalized, and use `bbox_space: "source_normalized"` or `"normalized"` when you want normalized interpretation explicitly.

`bbox_space` controls nested boxes:

- Omit it or use `"source"` when the box is in the original image coordinate space with auto pixel/normalized detection.
- Use `"source_pixels"` or `"pixels"` for absolute source-image pixel boxes.
- Use `"source_normalized"` or `"normalized"` for normalized source-image boxes.
- Use `"parent"` when a child box is relative to its parent crop with auto pixel/normalized detection. Normalized parent boxes are relative to the parent bbox width/height.
- Use `"parent_normalized"` / `"parent_relative"` for normalized child boxes relative to the parent crop.

`split_levels` is the default depth. The CLI `--levels` value overrides it.

The split script marks `target_leaf: true` on any node that reaches the
requested split depth, or any node with no children before that depth. In
component-first workflows, only target leaves become individual authoring
requests and source bundles before assembly.

Recommended fields:

- `id`: stable lowercase identifier using letters, digits, and underscores.
- `label`: human-readable component name.
- `bbox`: crop rectangle around the part.
- `role`: `scene`, `object`, `subassembly`, or `component`.
- `material_hint`: visible material or finish.
- `material_zone`: named region for planner material semantics.
- `function_hint`: what the part does mechanically or operationally.
- `signature_details`: shape-language details that must survive planning.
- `visual_facts`: typed silhouette, material-connectivity topology,
  negative-space, repetition, construction-boundary, and depth-order
  observations retained in the authoring request and source provenance.
  Repeated families include an integer `observed_count` only when centerline
  tracking through occluders supports one count. Otherwise set
  `observed_count: null`, `binding: false`, record sorted `plausible_counts`,
  and name a bounded integer `count_parameter` consumed by the linked pattern.
- `proportion_notes`: visible ratio/scale cues that should survive modeling.
- `construction_class`: dominant geometry route: `constant_thickness_planar_profile`,
  `primitive_prismatic_or_revolved`, `swept_section`, `lofted_transition`,
  `connected_member_assembly`, or `separate_body`.
- `profile_view`: for planar-profile parts, the source view normal to thickness
  from which the outer material loop and same-frame voids must be traced.
- `interface_notes`: mating faces, joint axes, holes, bosses, slots, rails, clearances, or contact patches.
- `visible_interfaces`: short interface list for planner and assembly checks.
- `expected_features`: short list of visible features the critic must look for.
- `crop_confidence`: `high`, `medium`, or `low`, with low-confidence notes in `cad_notes`.
- `detail_carrier`: `geometry`, `material`, `texture`, or `decal`.
- `cad_notes`: modeling observations, mating faces, symmetry, holes, clearances, or features to preserve.
- `children`: nested parts for level 2+ splits.

Every target leaf also produces:

- `authoring_brief_path`: provider-neutral brief with parts, parameters,
  interfaces, requested representations, and invariants;
- `source_manifest_path`: admitted `geometry.source.v1` component manifest;
- `local_frame`: origin and axis convention used for reassembly;
- `assembly_parent`, `contact_surfaces`, and optional typed `joint` record;
- `do_not_invent`: hidden or uncertain features the author must not add.

For dense products, split at mechanical boundaries: shell, panels, trays, handles, hinges, rails, knobs, grilles, tanks, rubber feet, fasteners, labels, displays, transparent lenses, and articulated parts.

Rigid does not mean monolithic. Repeated rails, slats, rods, fins, or bars that
terminate at a surrounding frame normally remain separate manufactured parts
when seams or butt joints are visible. Do not extend them into the frame simply
to manufacture positive-volume overlap in the model.

For scene-level prompts, use levels like this:

- Level 1: objects in the scene, for example machine, cabinet, conveyor cell, robot, vehicle.
- Level 2: parts inside each object, for example shell, panel, gripper, wheel, rail, tank.
- Level 3: subparts/hardware, for example screws, washers, clips, lenses, cable glands.

If the user specifies a number of levels, follow that number even when the default heuristic would choose a different depth.
