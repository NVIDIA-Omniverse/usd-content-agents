# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import math
import random
from collections.abc import Iterable, Iterator
from typing import Any

from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdLux, UsdShade, Vt

logger = logging.getLogger(__name__)


def traverse_prims(
    stage: Usd.Stage, traversal_method: str = "traverse_instanced_proxies"
) -> Iterator[Usd.Prim]:
    """Traverse all prims in a USD stage and yield them one at a time.

    Args:
        stage (Usd.Stage): The USD stage to traverse.
        traversal_method (str): The traversal method to use:
            "traverse" - Use Usd.PrimRange with default predicate
            "traverse_all" - Use Usd.PrimRange with PrimAllPrimsPredicate
            "traverse_instanced_proxies" - Use Usd.TraverseInstanceProxies

    Yields:
        Usd.Prim: Each prim found in the stage.
    """
    # Always start from the pseudo-root so that prims outside the default
    # prim hierarchy (e.g. IsaacSim UR10-style layouts) are included.
    start_prim = stage.GetPseudoRoot()

    if traversal_method == "traverse":
        # Use PrimRange with default predicate for consistent traversal
        for prim in Usd.PrimRange(start_prim):
            # Skip the pseudo-root prim ('/')
            if prim.IsPseudoRoot():
                continue
            yield prim
    elif traversal_method == "traverse_all":
        for prim in Usd.PrimRange(start_prim, Usd.PrimAllPrimsPredicate):
            # Skip the pseudo-root prim ('/')
            if prim.IsPseudoRoot():
                continue
            yield prim
    elif traversal_method == "traverse_instanced_proxies":
        for prim in Usd.PrimRange(start_prim, Usd.TraverseInstanceProxies()):
            # Skip the pseudo-root prim ('/')
            if prim.IsPseudoRoot():
                continue
            yield prim


def traverse_meshes(
    stage: Usd.Stage, traversal_method: str = "traverse_instanced_proxies"
) -> Iterator[UsdGeom.Mesh]:
    """Traverse all meshes in a USD stage and yield them one at a time.

    Args:
        stage (Usd.Stage): The USD stage to traverse.
        traversal_method (str): The traversal method to use:
            "traverse" - Use stage.Traverse()
            "traverse_all" - Use Usd.PrimRange with PrimAllPrimsPredicate
            "traverse_instanced_proxies" - Use Usd.TraverseInstanceProxies

    Yields:
        UsdGeom.Mesh: Each mesh found in the stage.
    """
    for prim in traverse_prims(stage, traversal_method):
        if prim.IsA(UsdGeom.Mesh):
            yield UsdGeom.Mesh(prim)


def get_all_mesh_prim_paths(
    stage: Usd.Stage, traversal_method: str = "traverse_instanced_proxies"
) -> list[str]:
    """Get all mesh prim paths in a USD stage.

    Args:
        stage (Usd.Stage): The USD stage to traverse.
        traversal_method (str): The traversal method to use:
            "traverse" - Use stage.Traverse()
            "traverse_all" - Use Usd.PrimRange with PrimAllPrimsPredicate
            "traverse_instanced_proxies" - Use Usd.TraverseInstanceProxies

    Returns:
        List[str]: The list of mesh prim paths.
    """
    prim_paths = []
    for mesh in traverse_meshes(stage, traversal_method):
        prim = mesh.GetPrim()
        prim_paths.append(prim.GetPath())
    return prim_paths


def collect_mesh_geometry_stats(
    stage: Usd.Stage,
    skip_geometry: bool = False,
    top_n: int = 10,
) -> dict[str, Any]:
    """Walk all prims, count types, optionally count vertices/faces per mesh.

    Traverses the full stage using ``stage.Traverse()`` (standard predicate)
    collecting prim-type counts, total mesh count, and (unless *skip_geometry*
    is set) per-mesh vertex and face counts with a top-N list.

    Args:
        stage: The USD stage to analyze.
        skip_geometry: If True, skip vertex/face counting (faster).
        top_n: Number of top meshes by vertex count to include.

    Returns:
        Dictionary with keys:
            - total_prims: int
            - total_meshes: int
            - prim_type_counts: dict[str, int] sorted by count descending
            - total_vertices: int  (only when skip_geometry=False)
            - total_faces: int  (only when skip_geometry=False)
            - top_meshes_by_vertices: list[dict]  (only when skip_geometry=False)
    """
    prim_type_counts: dict[str, int] = {}
    total_prims = 0
    total_meshes = 0
    total_vertices = 0
    total_faces = 0
    mesh_vertex_list: list[dict[str, Any]] = []

    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        total_prims += 1
        type_name = str(prim.GetTypeName()) if prim.GetTypeName() else "<no type>"
        prim_type_counts[type_name] = prim_type_counts.get(type_name, 0) + 1

        if type_name == "Mesh":
            total_meshes += 1
            if not skip_geometry:
                mesh = UsdGeom.Mesh(prim)
                points = mesh.GetPointsAttr().Get()
                face_counts = mesh.GetFaceVertexCountsAttr().Get()
                n_verts = len(points) if points else 0
                n_faces = len(face_counts) if face_counts else 0
                total_vertices += n_verts
                total_faces += n_faces
                mesh_vertex_list.append(
                    {
                        "path": str(prim.GetPath()),
                        "name": prim.GetName(),
                        "vertices": n_verts,
                        "faces": n_faces,
                    }
                )

    mesh_vertex_list.sort(key=lambda m: m["vertices"], reverse=True)

    result: dict[str, Any] = {
        "total_prims": total_prims,
        "total_meshes": total_meshes,
        "prim_type_counts": dict(sorted(prim_type_counts.items(), key=lambda x: -x[1])),
    }
    if not skip_geometry:
        result["total_vertices"] = total_vertices
        result["total_faces"] = total_faces
        result["top_meshes_by_vertices"] = mesh_vertex_list[:top_n]

    return result


def get_subtree_geometry_stats(
    stage: Usd.Stage,
    root_path: str,
    skip_geometry: bool = False,
) -> dict[str, Any]:
    """Get mesh/vertex/face counts and prim type breakdown for a subtree.

    Walks all prims under *root_path* (inclusive) using ``Usd.PrimRange``,
    counting meshes and optionally reading vertex/face data.

    Args:
        stage: The USD stage.
        root_path: Path of the subtree root prim.
        skip_geometry: If True, skip vertex/face counting.

    Returns:
        Dictionary with keys:
            - mesh_count: int
            - vertex_count: int
            - face_count: int
            - prim_type_breakdown: dict[str, int]
    """
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim or not root_prim.IsValid():
        return {
            "mesh_count": 0,
            "vertex_count": 0,
            "face_count": 0,
            "prim_type_breakdown": {},
        }

    mesh_count = 0
    vertex_count = 0
    face_count = 0
    type_counts: dict[str, int] = {}

    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        tn = str(prim.GetTypeName()) if prim.GetTypeName() else "<no type>"
        type_counts[tn] = type_counts.get(tn, 0) + 1
        if tn == "Mesh":
            mesh_count += 1
            if not skip_geometry:
                mesh = UsdGeom.Mesh(prim)
                pts = mesh.GetPointsAttr().Get()
                fcs = mesh.GetFaceVertexCountsAttr().Get()
                vertex_count += len(pts) if pts else 0
                face_count += len(fcs) if fcs else 0

    return {
        "mesh_count": mesh_count,
        "vertex_count": vertex_count,
        "face_count": face_count,
        "prim_type_breakdown": type_counts,
    }


def nullify_material(
    prim: Usd.Prim,
    set_triangle_winding_order: int = 0,
    exclude_list: list[str] | None = None,
    clear_ancestor_bindings: bool = True,
) -> None:
    """Nullify material from the prim.

    Removes material binding from the specified prim and blocks display color
    attributes. Can optionally set triangle winding order for meshes.

    Args:
        prim (Usd.Prim): The target prim.
        set_triangle_winding_order (int): Triangle winding order configuration:
            0 = no change
            1 = rightHanded
            2 = leftHanded
        exclude_list (list[str]): List of prim names to exclude from winding
            order changes.
        clear_ancestor_bindings (bool): If True, also clears material bindings
            from ancestor prims to prevent inheritance. Default: True.
    """
    if exclude_list is None:
        exclude_list = []

    # Check if there's actually a material binding to clear
    binding_api = UsdShade.MaterialBindingAPI(prim)
    if binding_api:
        rel = binding_api.GetDirectBindingRel()
        rel.SetTargets([])

    # Also clear material bindings from ancestor prims to prevent inheritance
    if clear_ancestor_bindings:
        parent = prim.GetParent()
        while parent and not parent.IsPseudoRoot():
            # Note: MaterialBindingAPI may be falsy if API not applied, but
            # GetDirectBindingRel() still works if the relationship exists
            parent_binding = UsdShade.MaterialBindingAPI(parent)
            parent_rel = parent_binding.GetDirectBindingRel()
            if parent_rel and parent_rel.GetTargets():
                parent_rel.SetTargets([])
            parent = parent.GetParent()
    # if prim.IsA(UsdGeom.Mesh):
    # mesh = UsdGeom.Mesh(prim)
    if prim.HasAttribute("primvars:displayColor"):
        attr = prim.GetAttribute("primvars:displayColor")
        primvar = UsdGeom.Primvar(attr)
        primvar.SetInterpolation("constant")  # Don't ask...
        attr.Block()

    if prim.HasAttribute("primvars:displayColor:indices"):
        attr = prim.GetAttribute("primvars:displayColor:indices")
        attr.Block()

    if prim.HasAttribute("primvars:displayOpacity"):
        attr = prim.GetAttribute("primvars:displayOpacity")
        attr.Block()

    if (
        set_triangle_winding_order != 0
        and prim.IsA(UsdGeom.Mesh)
        and (exclude_list is None or prim.GetName() not in exclude_list)
    ):
        mesh = UsdGeom.Mesh(prim)
        if set_triangle_winding_order == 1:
            mesh.CreateOrientationAttr().Set(UsdGeom.Tokens.rightHanded)
        else:
            mesh.CreateOrientationAttr().Set(UsdGeom.Tokens.leftHanded)
        # mesh.CreateDoubleSidedAttr().Set(True)


def nullify_materials(
    stage: Usd.Stage,
    prim_paths: list[str] | None = None,
    traversal_method: str = "traverse_instanced_proxies",
    set_triangle_winding_order: int = 0,
    exclude_list: list[str] | None = None,
) -> Usd.Stage:
    """Nullify materials for specific prims or entire stage.

    Removes material bindings from prims. Can operate in two modes:
    1. Targeted: Process only specified prim paths (efficient)
    2. Full traversal: Process all prims in stage (backward compatible)

    Args:
        stage (Usd.Stage): The USD stage to process.
        prim_paths (list[str] | None): List of prim paths to nullify materials for.
            If None or empty, traverses all prims in the stage (legacy behavior).
        traversal_method (str): The traversal method to use when prim_paths is empty:
            "traverse" - Use stage.Traverse()
            "traverse_all" - Use Usd.PrimRange with PrimAllPrimsPredicate
            "traverse_instanced_proxies" - Use Usd.TraverseInstanceProxies
        set_triangle_winding_order (int): Triangle winding order configuration:
            0 = no change
            1 = rightHanded
            2 = leftHanded
        exclude_list (list[str]): List of prim names to exclude from winding
            order changes.

    Returns:
        Usd.Stage: The updated USD stage.
    """
    if exclude_list is None:
        exclude_list = []

    # Determine which mode to use
    prim_iter: Iterable[Usd.Prim]
    if prim_paths:
        # Targeted mode: process only specified prims
        prim_iter = (stage.GetPrimAtPath(path) for path in prim_paths)
    else:
        # Full traversal mode: process all prims (backward compatible)
        prim_iter = traverse_prims(stage, traversal_method)

    processed_count = 0

    for prim in prim_iter:
        processed_count += 1

        if prim.IsInstanceProxy():
            continue

        if prim.IsInstance():
            prim.SetInstanceable(False)

        nullify_material(
            prim, set_triangle_winding_order, exclude_list if exclude_list else []
        )

    return stage


def set_gprim_display_color(
    gprim: UsdGeom.Gprim,
    color: tuple[float, float, float],
    time: Usd.TimeCode = Usd.TimeCode.Default(),
) -> None:
    """Set the display color of a geometric primitive.

    ``UsdGeom.Gprim`` is the common base for meshes and native geometry such as
    cubes, spheres, cylinders, cones, curves, and points. Using the base schema
    keeps render preparation type-agnostic while authoring the standard
    ``primvars:displayColor`` attribute.

    Args:
        gprim (UsdGeom.Gprim): The geometric primitive to color.
        color (Tuple[float, float, float]): RGB color values (0.0-1.0).
        time (Usd.TimeCode): The time to set the displayColor attribute.
    Returns:
        None

    Example:
        ```python
        cube = UsdGeom.Cube.Get(stage, "/path/to/cube")
        set_gprim_display_color(UsdGeom.Gprim(cube.GetPrim()), (1.0, 0.0, 0.0))
        ```
    """
    gprim.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]), time=time)


def set_mesh_display_color(
    mesh: UsdGeom.Mesh,
    color: tuple[float, float, float],
    time: Usd.TimeCode = Usd.TimeCode.Default(),
) -> None:
    """Set the display color of a mesh.

    This compatibility helper preserves the mesh-specific public API.
    """
    mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]), time=time)


def _is_valid_bbox_range(bbox_range: Gf.Range3d) -> bool:
    if bbox_range.IsEmpty():
        return False
    bbox_min = bbox_range.GetMin()
    bbox_max = bbox_range.GetMax()
    for i in range(3):
        if not math.isfinite(bbox_min[i]) or not math.isfinite(bbox_max[i]):
            return False
        if bbox_min[i] > bbox_max[i]:
            return False
    return True


def get_bbox_from_prim(
    prim: Usd.Prim,
    time: Usd.TimeCode = Usd.TimeCode.Default(),
    included_purposes: Iterable[str] | None = None,
) -> Gf.BBox3d:
    """Get the computed world-space bounding box (bbox) of a prim.

    Calculates and returns the actual bounding box for a prim in world space,
    not just the bounding box cache.

    Example:
    ```python
    # Create an in-memory stage
    stage = Usd.Stage.CreateInMemory()

    # Create a root transform and set as default prim
    root_prim = UsdGeom.Xform.Define(stage, '/Root')
    stage.SetDefaultPrim(root_prim.GetPrim())

    # Add a cube under the root
    cube_prim = UsdGeom.Cube.Define(stage, '/Root/Cube')

    # Get the world-space bounding box for the cube
    bbox = get_bbox_from_prim(cube_prim.GetPrim())

    # Print the bounding box information
    logger.info(f"Range: {bbox.GetRange()}")
    logger.info(f"Max: {bbox.GetMax()}")
    logger.info(f"Min: {bbox.GetMin()}")
    ```

    Args:
        prim (Usd.Prim): The prim for which to get the bbox.
        time: Time code used to compute the bound.
        included_purposes: Optional UsdGeom purposes to include. The legacy
            default includes only ``default`` geometry; callers must opt in to
            additional purposes such as ``render``.

    Returns:
        Gf.BBox3d: The computed world-space bounding box for the prim.
    """
    # Create a BBoxCache object to compute the bounding box
    bbox_cache = UsdGeom.BBoxCache(
        time,
        (
            list(included_purposes)
            if included_purposes is not None
            else [UsdGeom.Tokens.default_]
        ),
    )
    # useGeom = True, useInvisibleGeom = False

    # Compute the bounding box for the prim.
    bbox = bbox_cache.ComputeWorldBound(prim)
    aligned_range = bbox.ComputeAlignedRange()
    if _is_valid_bbox_range(aligned_range):
        return bbox

    # Some Boundable container prim types, notably UsdSkelRoot, can report an
    # empty own bound even when descendant meshes have valid bounds. Fall back to
    # an explicit descendant union so camera framing does not inherit USD's
    # empty-range sentinel values.
    bbox_min: Gf.Vec3d | None = None
    bbox_max: Gf.Vec3d | None = None
    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if child == prim:
            continue
        child_range = bbox_cache.ComputeWorldBound(child).ComputeAlignedRange()
        if not _is_valid_bbox_range(child_range):
            continue
        child_min = child_range.GetMin()
        child_max = child_range.GetMax()
        if bbox_min is None or bbox_max is None:
            bbox_min = Gf.Vec3d(child_min)
            bbox_max = Gf.Vec3d(child_max)
        else:
            bbox_min = Gf.Vec3d(
                min(bbox_min[0], child_min[0]),
                min(bbox_min[1], child_min[1]),
                min(bbox_min[2], child_min[2]),
            )
            bbox_max = Gf.Vec3d(
                max(bbox_max[0], child_max[0]),
                max(bbox_max[1], child_max[1]),
                max(bbox_max[2], child_max[2]),
            )

    if bbox_min is not None and bbox_max is not None:
        return Gf.BBox3d(Gf.Range3d(bbox_min, bbox_max))

    logger.warning(
        "No valid world-space bbox for prim %s at time %s; using zero-size fallback.",
        prim.GetPath(),
        time,
    )
    origin = Gf.Vec3d(0.0, 0.0, 0.0)
    return Gf.BBox3d(Gf.Range3d(origin, origin))


def print_prim_hierarchy(stage: Usd.Stage) -> None:
    """Print a tree-like structure of all prims in the USD stage.

    Creates a visual tree representation of the USD stage hierarchy using ASCII
    characters to show parent-child relationships. This makes it easier to
    visualize complex hierarchies compared to simple indentation.

    The output format uses:
    - ├── for branches
    - └── for the last item in a branch
    - │ for vertical connections

    Args:
        stage (Usd.Stage): USD stage to print the hierarchy for.

    Returns:
        None

    Example:
        ```python
        stage = Usd.Stage.Open("mymodel.usd")
        print_prim_hierarchy(stage)
        # Output might look like:
        # Root
        # ├── Materials
        # │   ├── Red
        # │   └── Blue
        # └── Geometry
        #     ├── Sphere
        #     └── Cube
        ```
    """

    def _print_prim_recursive(prim: Usd.Prim, prefix: str = "") -> None:
        """Helper function to recursively print prim hierarchy.

        Args:
            prim (Usd.Prim): The current prim to process.
            prefix (str): Prefix string for tree-like output formatting.

        Returns:
            None
        """
        # Get all direct children of this prim
        children = list(prim.GetChildren())

        # Process each child
        for i, child in enumerate(children):
            is_last = i == len(children) - 1

            # Print current prim with appropriate tree characters
            if is_last:
                print(f"{prefix}└── {child.GetName()}")
                next_prefix = prefix + "    "
            else:
                print(f"{prefix}├── {child.GetName()}")
                next_prefix = prefix + "│   "

            # Recursively process child's children
            _print_prim_recursive(child, next_prefix)

    # Get the pseudo-root prim and print it
    root = stage.GetPseudoRoot()
    print(root.GetName() or "Root")

    # Start the recursive traversal from the pseudo-root
    _print_prim_recursive(root)


def remove_scope_and_prims_under_it(stage: Usd.Stage, scope_path: str) -> None:
    """Remove a scope and all prims under it.

    Removes a prim at the specified path and all its descendant prims from
    the USD stage. This function first checks if the scope prim exists, then
    collects all prims under it, and finally removes them from the stage.

    Args:
        stage (Usd.Stage): The USD stage.
        scope_path (str): The path to the scope to remove.

    Returns:
        None

    Example:
        ```python
        stage = Usd.Stage.Open("my_stage.usd")
        remove_scope_and_prims_under_it(stage, "/Root/Group1")
        ```
    """
    scope_prim = stage.GetPrimAtPath(scope_path)
    prims_to_remove: list[Usd.Prim] = []
    if scope_prim.IsValid():
        for prim in Usd.PrimRange(scope_prim):
            prims_to_remove.append(prim)

    for prim in prims_to_remove:
        stage.RemovePrim(prim.GetPath())


def remove_all_lights(stage: Usd.Stage) -> None:
    """Remove all light prims from a USD stage.

    Uses RemovePrim which writes to the root layer and persists through
    GetRootLayer().Export() (used by NVCF). Falls back to SetActive(False)
    if RemovePrim fails (e.g. read-only layers like USDZ archives).

    Args:
        stage (Usd.Stage): The USD stage to remove light prims from.

    Returns:
        None

    Example:
        ```python
        stage = Usd.Stage.Open("scene_with_lights.usd")
        remove_all_lights(stage)
        stage.Save()  # Save the modified stage
        ```
    """
    light_paths: list[str] = [
        str(prim.GetPath())
        for prim in traverse_prims(stage)
        if prim.HasAPI(UsdLux.LightAPI) and not prim.IsInstanceProxy()
    ]

    # Remove lights from the stage.
    # Use RemovePrim which writes to the root layer and persists through
    # GetRootLayer().Export() (used by NVCF). Falls back to SetActive(False)
    # if RemovePrim fails (e.g. read-only layers).
    for path in light_paths:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsActive():
            if not stage.RemovePrim(path):
                prim.SetActive(False)


def assign_color_to_meshes(
    stage: Usd.Stage,
    color: tuple[float, float, float],
    traversal_method: str = "traverse_instanced_proxies",
    time: Usd.TimeCode = Usd.TimeCode.Default(),
) -> None:
    """Assign a specific color to all meshes in a USD stage.

    Iterates through all prims in the stage and assigns the specified color to
    each mesh prim. The color is stored as a displayColor attribute.

    Args:
        stage (Usd.Stage): The USD stage to assign a color to.
        color (Tuple[float, float, float]): RGB color values (0.0-1.0).
        traversal_method (str): The traversal method to use:
            "traverse" - Use stage.Traverse()
            "traverse_all" - Use Usd.PrimRange with PrimAllPrimsPredicate
            "traverse_instanced_proxies" - Use Usd.TraverseInstanceProxies
        time (Usd.TimeCode): The time to set the displayColor attribute.
    Returns:
        None

    Example:
        ```python
        stage = Usd.Stage.Open("model.usd")
        assign_color_to_meshes(stage, (1.0, 0.0, 0.0))
        stage.Save()
        ```
    """
    # Get all prims in the stage
    prims = traverse_prims(stage, traversal_method)

    # Iterate over prims and assign the specified color to meshes
    for prim in prims:
        if prim.IsA(UsdGeom.Mesh):
            # Skip instance proxies - they cannot be edited directly
            if prim.IsInstanceProxy():
                continue
            # Make instances non-instanceable so we can edit them
            if prim.IsInstance():
                prim.SetInstanceable(False)
            mesh = UsdGeom.Mesh(prim)
            set_mesh_display_color(mesh, color, time=time)


def assign_random_colors_to_meshes(
    stage: Usd.Stage,
    traversal_method: str = "traverse_instanced_proxies",
    range_min: float = 0.5,
    range_max: float = 0.75,
    time: Usd.TimeCode = Usd.TimeCode.Default(),
) -> None:
    """Assign random pastel colors to all meshes in a USD stage.

    Iterates through all prims in the stage and assigns a random pastel color
    to each mesh prim. The color is stored as a displayColor attribute on the
    mesh.

    Args:
        stage (Usd.Stage): The USD stage to assign random colors to.
        traversal_method (str): The traversal method to use:
            "traverse" - Use stage.Traverse()
            "traverse_all" - Use Usd.PrimRange with PrimAllPrimsPredicate
            "traverse_instanced_proxies" - Use Usd.TraverseInstanceProxies
        range_min (float): The minimum value for the color range.
        range_max (float): The maximum value for the color range.
        time (Usd.TimeCode): The time to set the displayColor attribute.
    Returns:
        None

    Example:
        ```python
        stage = Usd.Stage.Open("model.usd")
        assign_random_colors_to_meshes(stage, time=Usd.TimeCode.Default())
        stage.Save()
        ```
    """
    # Get all prims in the stage
    prims = traverse_prims(stage, traversal_method)

    # Iterate over prims and assign random pastel colors to meshes
    for prim in prims:
        if prim.IsA(UsdGeom.Mesh):
            # Skip instance proxies - they cannot be edited directly
            if prim.IsInstanceProxy():
                continue
            # Make instances non-instanceable so we can edit them
            if prim.IsInstance():
                prim.SetInstanceable(False)

            mesh = UsdGeom.Mesh(prim)

            # Generate a random pastel color
            # Pastel colors have high values (0.5-1.0) in RGB
            pastel_color = (
                random.uniform(range_min, range_max),  # R
                random.uniform(range_min, range_max),  # G
                random.uniform(range_min, range_max),  # B
            )

            # Apply the color to the mesh
            set_mesh_display_color(mesh, pastel_color, time=time)


def convert_abstract_prototypes_to_def(
    stage: Usd.Stage,
    prototype_names: list[str] | None = None,
) -> int:
    """Convert abstract prototype prims (class/over) to concrete def prims IN-PLACE.

    This function directly modifies prim specifiers in the layer, preserving all
    references, transforms, and other composition arcs. It simply changes 'class'
    or 'over' specifiers to 'def' for matching prototype prims.

    Handles two common scenarios:
    - 'class' prototypes: Abstract prims used for inheritance (e.g., class Scope "Prototypes")
    - 'over' prototypes: Override prims in flattened files (e.g., over "Flattened_Prototype_11")

    Args:
        stage (Usd.Stage): The USD stage to process.
        prototype_names (list[str] | None): List of specific prim names to convert.
            If None, converts all prims with "Prototype" in their name.

    Returns:
        int: Number of abstract prims converted to def.

    Example:
        # Convert all prototypes (class or over) to def
        count = convert_abstract_prototypes_to_def(stage)

        # Convert specific prototypes only
        count = convert_abstract_prototypes_to_def(
            stage, prototype_names=["Prototypes", "Flattened_Prototype_11"]
        )
    """
    converted_count = 0

    # Get the root layer for direct modification
    layer = stage.GetRootLayer()

    def _convert_prim_spec_recursive(prim_spec: Sdf.PrimSpec) -> int:
        """Recursively convert abstract prim specs to def in-place."""
        count = 0
        prim_name = prim_spec.name

        # Check if this prim should be converted
        should_convert = False
        if prim_spec.specifier in (Sdf.SpecifierClass, Sdf.SpecifierOver):
            if prototype_names is None:
                # Convert if name contains "Prototype" (case-insensitive)
                should_convert = "prototype" in prim_name.lower()
            else:
                # Convert if name matches any in the specified list
                should_convert = prim_name in prototype_names

        if should_convert:
            specifier_name = (
                "class" if prim_spec.specifier == Sdf.SpecifierClass else "over"
            )
            # Change specifier to def
            prim_spec.specifier = Sdf.SpecifierDef

            # Set typeName if empty (required for def prims to be valid)
            if not prim_spec.typeName:
                prim_spec.typeName = "Xform"

            logger.info(
                f"Converting {specifier_name} '{prim_name}' to def at {prim_spec.path}"
            )
            count += 1

        # Recursively process children
        for child_spec in prim_spec.nameChildren:
            count += _convert_prim_spec_recursive(child_spec)

        return count

    try:
        # Process all root prims
        for prim_spec in layer.rootPrims:
            converted_count += _convert_prim_spec_recursive(prim_spec)

    except Exception:
        logger.exception("Failed to convert prototypes")
        raise

    if converted_count > 0:
        logger.info(
            f"Successfully converted {converted_count} abstract prototype(s) to def in-place"
        )
    else:
        logger.info("No abstract prototypes found to convert")

    return converted_count


# Backward compatibility alias
convert_class_prototypes_to_xforms = convert_abstract_prototypes_to_def


def flatten_prototype_references(
    stage: Usd.Stage,
    remove_prototypes: bool = True,
) -> Sdf.Layer:
    """Flatten a stage by copying composed values, resolving all references.

    This function creates a fully flattened layer where:
    1. All references are resolved by copying composed attribute values
    2. Prototype prims (Flattened_Prototype_*) are excluded from output
    3. Each instance becomes self-contained with correct transforms and geometry

    This is useful when you want to run Scene Optimizer on a pre-flattened file,
    as the optimizer works better with concrete geometry rather than references.

    Args:
        stage: The USD stage to flatten.
        remove_prototypes: If True, exclude Flattened_Prototype_* prims from output.

    Returns:
        A new Sdf.Layer with flattened content.

    Example:
        stage = Usd.Stage.Open("input.usd")
        flattened_layer = flatten_prototype_references(stage)
        flattened_layer.Export("flattened.usda")
    """
    logger.info("Flattening stage by copying composed values...")

    # Create output layer
    output_layer = Sdf.Layer.CreateAnonymous("flattened")

    # Copy stage-level metadata from source layer
    source_layer = stage.GetRootLayer()

    # Copy pseudoroot metadata (upAxis, metersPerUnit, etc.)
    source_pseudoroot = source_layer.pseudoRoot
    target_pseudoroot = output_layer.pseudoRoot

    # Copy layer metadata
    if source_layer.HasDefaultPrim():
        output_layer.defaultPrim = source_layer.defaultPrim
    if source_layer.documentation:
        output_layer.documentation = source_layer.documentation

    # Copy pseudoroot custom data and metadata
    for key in source_pseudoroot.customData:
        target_pseudoroot.customData[key] = source_pseudoroot.customData[key]

    # Copy stage metadata like upAxis, metersPerUnit via UsdGeom
    up_axis = UsdGeom.GetStageUpAxis(stage)
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)

    # These need to be set on the pseudoroot's customData
    if up_axis:
        target_pseudoroot.SetInfo("upAxis", up_axis)
    # Always preserve metersPerUnit — skipping the default (0.01) caused
    # the value to be silently lost when the source stage used 1.0 (meters).
    target_pseudoroot.SetInfo("metersPerUnit", meters_per_unit)

    def _should_skip_path(prim_path: Sdf.Path) -> bool:
        """Check if prim path should be skipped (prototype paths)."""
        if not remove_prototypes:
            return False
        path_str = str(prim_path)

        # Skip root-level Flattened_Prototype_* prims and anything under them
        parts = path_str.split("/")
        if len(parts) >= 2:
            root_name = parts[1]
            if "Flattened_Prototype" in root_name:
                return True

        # Skip any prim under a /Prototypes/ scope (nested prototype containers)
        if "/Prototypes/" in path_str or path_str.endswith("/Prototypes"):
            return True

        return False

    def _ensure_parent_chain(prim_path: Sdf.Path, target_layer: Sdf.Layer) -> None:
        """Ensure all parent prims exist in the target layer."""
        parent_path = prim_path.GetParentPath()
        if parent_path == Sdf.Path.absoluteRootPath or parent_path.isEmpty:
            return

        if target_layer.GetPrimAtPath(parent_path):
            return  # Parent already exists

        # Recursively ensure grandparents exist
        _ensure_parent_chain(parent_path, target_layer)

        # Create parent prim
        grandparent_path = parent_path.GetParentPath()
        prim_name = parent_path.name

        if grandparent_path == Sdf.Path.absoluteRootPath or grandparent_path.isEmpty:
            # Root-level prim
            Sdf.PrimSpec(target_layer, prim_name, Sdf.SpecifierDef)
        else:
            parent_spec = target_layer.GetPrimAtPath(grandparent_path)
            if parent_spec:
                Sdf.PrimSpec(parent_spec, prim_name, Sdf.SpecifierDef)

    def _copy_prim_to_layer(
        composed_prim: Usd.Prim, target_layer: Sdf.Layer
    ) -> Sdf.PrimSpec | None:
        """Copy a composed prim to the target layer."""
        prim_path = composed_prim.GetPath()

        # Skip if already exists
        existing = target_layer.GetPrimAtPath(prim_path)
        if existing:
            return existing

        # Ensure parent chain exists
        _ensure_parent_chain(prim_path, target_layer)

        # Create prim spec
        parent_path = prim_path.GetParentPath()
        prim_name = prim_path.name

        if parent_path == Sdf.Path.absoluteRootPath or parent_path.isEmpty:
            prim_spec = Sdf.PrimSpec(target_layer, prim_name, Sdf.SpecifierDef)
        else:
            parent_spec = target_layer.GetPrimAtPath(parent_path)
            if not parent_spec:
                logger.warning(f"Parent not found for {prim_path}")
                return None
            prim_spec = Sdf.PrimSpec(parent_spec, prim_name, Sdf.SpecifierDef)

        # Set type
        type_name = composed_prim.GetTypeName()
        if type_name:
            prim_spec.typeName = type_name

        # Copy key prim metadata explicitly (Usd.Prim doesn't expose all metadata keys)
        # Keep this list small to avoid composition arc issues.
        metadata_keys = [
            "apiSchemas",
            "kind",
            "purpose",
            "instanceable",
            "active",
            "hidden",
        ]
        for key in metadata_keys:
            value = composed_prim.GetMetadata(key)
            if value is not None:
                prim_spec.SetInfo(key, value)

        # Copy customData explicitly (may include apiSchemas-related data)
        custom_data = composed_prim.GetCustomData()
        if custom_data:
            for key, value in custom_data.items():
                # Skip keys that are not valid USD identifiers (e.g., "3dsmax"
                # from 3ds Max exports — keys starting with a digit are invalid)
                if not Sdf.Path.IsValidIdentifier(key):
                    logger.debug(
                        "Skipping invalid customData key %r on %s", key, prim_path
                    )
                    continue
                prim_spec.customData[key] = value

        # Copy attributes from composed prim (values, connections, and shader IO)
        for attr in composed_prim.GetAttributes():
            has_value = attr.HasValue()
            value_is_blocked = attr.GetResolveInfo().ValueIsBlocked()
            connections = attr.GetConnections()
            attr_name = attr.GetName()
            is_shader_io = attr_name.startswith("inputs:") or attr_name.startswith(
                "outputs:"
            )

            if has_value or value_is_blocked or connections or is_shader_io:
                try:
                    attr_spec = Sdf.AttributeSpec(
                        prim_spec,
                        attr_name,
                        attr.GetTypeName(),
                        attr.GetVariability(),
                        attr.IsCustom(),
                    )

                    # Structural fields are supplied to AttributeSpec above. Copy
                    # the remaining authored metadata, including primvar
                    # interpolation and elementSize.
                    for key, value in attr.GetAllAuthoredMetadata().items():
                        if key not in {
                            "connectionPaths",
                            "custom",
                            "targetPaths",
                            "typeName",
                            "variability",
                        }:
                            attr_spec.SetInfo(key, value)

                    # A blocked indices attribute is how an authored primvar can
                    # explicitly remain unindexed. Do not silently discard it.
                    if value_is_blocked:
                        attr_spec.default = Sdf.ValueBlock()
                    elif has_value:
                        val = attr.Get()
                        if val is not None:
                            attr_spec.default = val

                    # Copy connections (for shader networks)
                    if connections:
                        for conn_path in connections:
                            # Filter out connections to prototype paths
                            if not _should_skip_path(conn_path):
                                attr_spec.connectionPathList.Append(conn_path)

                except (Tf.ErrorException, RuntimeError) as e:
                    logger.debug(f"Skipping attribute {attr.GetName()}: {e}")

        # Copy relationships
        for rel in composed_prim.GetRelationships():
            targets = rel.GetTargets()
            if targets:
                try:
                    rel_spec = Sdf.RelationshipSpec(prim_spec, rel.GetName(), False)
                    for target in targets:
                        # Filter out targets that point to prototype paths
                        if not _should_skip_path(target):
                            rel_spec.targetPathList.Append(target)
                except (Tf.ErrorException, RuntimeError) as e:
                    logger.debug(f"Skipping relationship {rel.GetName()}: {e}")

        return prim_spec

    # Use Traverse() with TraverseInstanceProxies to get full composed hierarchy
    # This includes children that come from references
    prim_count = 0
    for composed_prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        prim_path = composed_prim.GetPath()

        # Skip prototype paths
        if _should_skip_path(prim_path):
            continue

        _copy_prim_to_layer(composed_prim, output_layer)
        prim_count += 1

    logger.info(f"Flattening complete - copied {prim_count} prims")
    return output_layer


def _copy_layer_metadata(source_layer: Sdf.Layer, target_layer: Sdf.Layer) -> None:
    """Copy layer-level metadata from source to target layer."""
    # Copy pseudoRoot metadata
    pseudoroot_source = source_layer.pseudoRoot
    pseudoroot_target = target_layer.pseudoRoot

    # Copy all metadata from pseudoroot
    for key in pseudoroot_source.GetMetaDataInfoKeys():
        value = pseudoroot_source.GetInfo(key)
        if value is not None:
            pseudoroot_target.SetInfo(key, value)


def _copy_prim_spec_recursive(
    source_spec: Sdf.PrimSpec,
    target_layer: Sdf.Layer,
    target_parent_path: Sdf.Path | None,
    prototype_names: list[str] | None,
) -> int:
    """Recursively copy a prim spec and all its children, converting abstract prototypes.

    Handles both 'class' and 'over' specifiers for prototype prims, converting them
    to concrete 'def' prims so they become traversable in the composed stage.

    Returns:
        int: Number of abstract prototypes converted in this subtree.
    """
    converted_count = 0

    # Determine if this should be converted (handles both 'class' and 'over' specifiers)
    should_convert = False
    prim_name = source_spec.name
    if source_spec.specifier in (Sdf.SpecifierClass, Sdf.SpecifierOver):
        if prototype_names is None:
            # Convert if name contains "Prototype" (case-insensitive)
            should_convert = "prototype" in prim_name.lower()
        else:
            # Convert if name matches any in the specified list
            should_convert = prim_name in prototype_names

    # Create target path
    if target_parent_path is None:
        target_path = Sdf.Path(f"/{source_spec.name}")
    else:
        target_path = target_parent_path.AppendChild(source_spec.name)

    # Create the new prim spec
    # For root prims, pass the layer; for child prims, pass the parent spec
    if target_parent_path is None:
        target_spec = Sdf.PrimSpec(target_layer, source_spec.name, Sdf.SpecifierDef)
    else:
        target_parent_spec = target_layer.GetPrimAtPath(target_parent_path)
        target_spec = Sdf.PrimSpec(
            target_parent_spec, source_spec.name, Sdf.SpecifierDef
        )

    # Set specifier and type name
    if should_convert:
        target_spec.specifier = Sdf.SpecifierDef
        # Preserve existing typeName for 'over' prims, use Xform for 'class' or if no type
        if source_spec.typeName:
            target_spec.typeName = source_spec.typeName
        else:
            target_spec.typeName = "Xform"
        converted_count += 1
        specifier_name = (
            "class" if source_spec.specifier == Sdf.SpecifierClass else "over"
        )
        logger.info(
            f"Converting {specifier_name} '{source_spec.name}' to def at {target_path}"
        )
    else:
        target_spec.specifier = source_spec.specifier
        # Only set typeName if non-empty (USD doesn't allow setting empty type names)
        if source_spec.typeName:
            target_spec.typeName = source_spec.typeName

    # Copy all other metadata and info
    for key in source_spec.GetMetaDataInfoKeys():
        if key not in ("specifier", "typeName"):  # Skip what we already handled
            value = source_spec.GetInfo(key)
            if value is not None:
                target_spec.SetInfo(key, value)

    # Copy properties
    for prop_spec in source_spec.properties:
        _copy_property_spec(prop_spec, target_spec)

    # Recursively copy children
    for child_spec in source_spec.nameChildren:
        child_converted = _copy_prim_spec_recursive(
            child_spec, target_layer, target_path, prototype_names
        )
        converted_count += child_converted

    return converted_count


def _copy_property_spec(
    source_prop: Sdf.PropertySpec, target_prim: Sdf.PrimSpec
) -> None:
    """Copy a property spec to the target prim spec."""
    if isinstance(source_prop, Sdf.AttributeSpec):
        # Copy attribute
        target_attr = Sdf.AttributeSpec(
            target_prim, source_prop.name, source_prop.typeName, source_prop.variability
        )

        # Copy the default value (e.g., points, normals arrays for Mesh)
        if source_prop.HasDefaultValue():
            target_attr.default = source_prop.default

        # Copy all attribute info/metadata
        for key in source_prop.GetMetaDataInfoKeys():
            value = source_prop.GetInfo(key)
            if value is not None:
                target_attr.SetInfo(key, value)

        # Copy time samples if any
        layer = source_prop.layer
        attr_path = source_prop.path
        time_samples = layer.ListTimeSamplesForPath(attr_path)
        if time_samples:
            target_layer = target_prim.layer
            for time in time_samples:
                value = layer.QueryTimeSample(attr_path, time)
                target_layer.SetTimeSample(target_attr.path, time, value)

    elif isinstance(source_prop, Sdf.RelationshipSpec):
        # Copy relationship
        # RelationshipSpec signature: (ownerPrim, name, custom, variability)
        target_rel = Sdf.RelationshipSpec(
            target_prim,
            source_prop.name,
            source_prop.custom,
            source_prop.variability,
        )

        # Copy all relationship info
        for key in source_prop.GetMetaDataInfoKeys():
            value = source_prop.GetInfo(key)
            if value is not None:
                target_rel.SetInfo(key, value)

        # Copy target paths for relationships
        for target_path in source_prop.targetPathList.GetAddedOrExplicitItems():
            target_rel.targetPathList.Append(target_path)


def enable_visibility_for_all_mesh_prims(
    stage: Usd.Stage,
    traversal_method: str = "traverse_instanced_proxies",
    time: Usd.TimeCode = Usd.TimeCode.Default(),
) -> None:
    """Enable visibility for all mesh prims in a USD stage.

    Iterates through all prims in the stage and sets their visibility attribute
    to visible.

    Args:
        stage (Usd.Stage): The USD stage to modify.
        traversal_method (str): The traversal method to use:
            "traverse" - Use stage.Traverse()
            "traverse_all" - Use Usd.PrimRange with PrimAllPrimsPredicate
            "traverse_instanced_proxies" - Use Usd.TraverseInstanceProxies
        time (Usd.TimeCode): The time to set the visibility attribute.

    Returns:
        None

    Example:
        ```python
        stage = Usd.Stage.Open("model.usd")
        enable_visibility_for_all_mesh_prims(stage)
        ```
    """
    # Get all prims in the stage
    prims = traverse_prims(stage, traversal_method)

    # Iterate over prims and enable visibility
    for prim in prims:
        if prim.IsA(UsdGeom.Mesh):
            # Skip instance proxies - they cannot be edited directly
            if prim.IsInstanceProxy():
                continue
            # Make instances non-instanceable so we can edit them
            if prim.IsInstance():
                prim.SetInstanceable(False)

            mesh = UsdGeom.Mesh(prim)
            mesh.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited, time=time)


def enable_visibility_except_for_selected_mesh_prims(
    stage: Usd.Stage,
    prim_paths: list[str],
    traversal_method: str = "traverse_instanced_proxies",
    time: Usd.TimeCode = Usd.TimeCode.Default(),
) -> None:
    """Enable visibility for all mesh prims except the selected prims.

    Args:
        stage (Usd.Stage): The USD stage to modify.
        prim_paths (List[str]): The paths of the prims to keep visible.
        traversal_method (str): The traversal method to use:
            "traverse" - Use stage.Traverse()
            "traverse_all" - Use Usd.PrimRange with PrimAllPrimsPredicate
            "traverse_instanced_proxies" - Use Usd.TraverseInstanceProxies
        time (Usd.TimeCode): The time to set the visibility attribute.

    Returns:
        None

    Example:
        ```python
        stage = Usd.Stage.Open("model.usd")
        enable_visibility_except_for_selected_mesh_prims(
            stage, ["/Root/SelectedPrim1", "/Root/SelectedPrim2"]
        )
        ```
    """
    # Get all prims in the stage
    prims = traverse_prims(stage, traversal_method)

    # Iterate over prims and enable visibility for all except selected prims
    for prim in prims:
        if prim.GetPath().pathString not in prim_paths:
            if prim.IsA(UsdGeom.Mesh):
                # Skip instance proxies - they cannot be edited directly
                if prim.IsInstanceProxy():
                    continue
                # Make instances non-instanceable so we can edit them
                if prim.IsInstance():
                    prim.SetInstanceable(False)

                mesh = UsdGeom.Mesh(prim)
                mesh.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited, time=time)


def disable_visibility_for_all_gprims(
    stage: Usd.Stage,
    traversal_method: str = "traverse_instanced_proxies",
    time: Usd.TimeCode = Usd.TimeCode.Default(),
) -> None:
    """Disable visibility for all editable ``UsdGeom.Gprim`` prims.

    This includes meshes and native geometric primitives. USD does not permit
    authoring visibility on instance proxies directly, so remaining proxy
    geometry is hidden through its nearest editable instance root. This keeps
    unrelated instances intact while preserving prim-only render isolation.
    """
    gprim_paths = [
        str(prim.GetPath())
        for prim in traverse_prims(stage, traversal_method)
        if prim.IsA(UsdGeom.Gprim)
    ]
    hidden_instance_roots: set[str] = set()

    for prim_path in gprim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Gprim):
            continue

        masked_by_instance_root = False
        deinstanced_roots: set[str] = set()
        while prim.IsInstanceProxy():
            instance_root = prim
            while (
                instance_root
                and instance_root.IsValid()
                and not instance_root.IsPseudoRoot()
                and (not instance_root.IsInstance() or instance_root.IsInstanceProxy())
            ):
                instance_root = instance_root.GetParent()

            if (
                not instance_root
                or not instance_root.IsValid()
                or not instance_root.IsInstance()
                or instance_root.IsInstanceProxy()
            ):
                raise ValueError(
                    f"Gprim instance proxy has no editable instance root: {prim_path}"
                )

            instance_root_path = str(instance_root.GetPath())
            imageable_root = UsdGeom.Imageable(instance_root)
            if imageable_root:
                if instance_root_path not in hidden_instance_roots:
                    imageable_root.GetVisibilityAttr().Set(
                        UsdGeom.Tokens.invisible,
                        time=time,
                    )
                    hidden_instance_roots.add(instance_root_path)
                masked_by_instance_root = True
                break

            if instance_root_path in deinstanced_roots:
                raise ValueError(
                    "Gprim remained an instance proxy after de-instancing "
                    f"{instance_root_path}: {prim_path}"
                )
            deinstanced_roots.add(instance_root_path)
            instance_root.SetInstanceable(False)
            prim = stage.GetPrimAtPath(prim_path)

        if masked_by_instance_root:
            continue
        if prim.IsInstance():
            prim.SetInstanceable(False)
            prim = stage.GetPrimAtPath(prim_path)
        UsdGeom.Gprim(prim).GetVisibilityAttr().Set(
            UsdGeom.Tokens.invisible,
            time=time,
        )


def disable_visibility_for_all_mesh_prims(
    stage: Usd.Stage,
    traversal_method: str = "traverse_instanced_proxies",
    time: Usd.TimeCode = Usd.TimeCode.Default(),
) -> None:
    """Disable visibility for all mesh prims in a USD stage.

    Iterates through all prims in the stage and sets their visibility attribute
    to invisible.

    Args:
        stage (Usd.Stage): The USD stage to modify.
        traversal_method (str): The traversal method to use:
            "traverse" - Use stage.Traverse()
            "traverse_all" - Use Usd.PrimRange with PrimAllPrimsPredicate
            "traverse_instanced_proxies" - Use Usd.TraverseInstanceProxies
    Returns:
        None

    Example:
        ```python
        stage = Usd.Stage.Open("model.usd")
        disable_visibility_for_all_mesh_prims(stage)
        ```
    """
    # Get all prims in the stage
    prims = traverse_prims(stage, traversal_method)

    # Iterate over prims and disable visibility
    for prim in prims:
        if prim.IsA(UsdGeom.Mesh):
            # Skip instance proxies - they cannot be edited directly
            if prim.IsInstanceProxy():
                continue
            # Make instances non-instanceable so we can edit them
            if prim.IsInstance():
                prim.SetInstanceable(False)

            mesh = UsdGeom.Mesh(prim)
            mesh.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible, time=time)


def disable_visibility_except_for_selected_mesh_prim(
    stage: Usd.Stage,
    prim_path: str,
    traversal_method: str = "traverse_instanced_proxies",
    time: Usd.TimeCode = Usd.TimeCode.Default(),
) -> None:
    """Disable visibility for all mesh prims except the selected prim.

    Args:
        stage (Usd.Stage): The USD stage to modify.
        prim_path (str): The path of the prim to keep visible.
        traversal_method (str): The traversal method to use:
            "traverse" - Use stage.Traverse()
            "traverse_all" - Use Usd.PrimRange with PrimAllPrimsPredicate
            "traverse_instanced_proxies" - Use Usd.TraverseInstanceProxies
        time (Usd.TimeCode): The time to set the visibility attribute.
    Returns:
        None

    Example:
        ```python
        stage = Usd.Stage.Open("model.usd")
        disable_visibility_except_for_selected_mesh_prim(
            stage, "/Root/SelectedPrim"
        )
        ```
    """
    # Get all prims in the stage
    prims = traverse_prims(stage, traversal_method)

    # Iterate over prims and disable visibility for all except selected prim
    for prim in prims:
        if prim.GetPath().pathString != prim_path:
            if prim.IsA(UsdGeom.Mesh):
                # Skip instance proxies - they cannot be edited directly
                if prim.IsInstanceProxy():
                    continue
                # Make instances non-instanceable so we can edit them
                if prim.IsInstance():
                    prim.SetInstanceable(False)

                mesh = UsdGeom.Mesh(prim)
                mesh.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible, time=time)


def disable_visibility_except_for_selected_mesh_prims(
    stage: Usd.Stage,
    prim_paths: list[str],
    traversal_method: str = "traverse_instanced_proxies",
    time: Usd.TimeCode = Usd.TimeCode.Default(),
) -> None:
    """Disable visibility for all mesh prims except the selected prims.

    Args:
        stage (Usd.Stage): The USD stage to modify.
        prim_paths (List[str]): The paths of the prims to keep visible.
        traversal_method (str): The traversal method to use:
            "traverse" - Use stage.Traverse()
            "traverse_all" - Use Usd.PrimRange with PrimAllPrimsPredicate
            "traverse_instanced_proxies" - Use Usd.TraverseInstanceProxies
        time (Usd.TimeCode): The time to set the visibility attribute.

    Returns:
        None

    Example:
        ```python
        stage = Usd.Stage.Open("model.usd")
        disable_visibility_except_for_selected_mesh_prims(
            stage, ["/Root/SelectedPrim1", "/Root/SelectedPrim2"]
        )
        ```
    """
    # Get all prims in the stage
    prims = traverse_prims(stage, traversal_method)

    # Iterate over prims and disable visibility for all except selected prims
    for prim in prims:
        if prim.GetPath().pathString not in prim_paths:
            if prim.IsA(UsdGeom.Mesh):
                # Skip instance proxies - they cannot be edited directly
                if prim.IsInstanceProxy():
                    continue
                # Make instances non-instanceable so we can edit them
                if prim.IsInstance():
                    prim.SetInstanceable(False)

                mesh = UsdGeom.Mesh(prim)
                mesh.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible, time=time)
