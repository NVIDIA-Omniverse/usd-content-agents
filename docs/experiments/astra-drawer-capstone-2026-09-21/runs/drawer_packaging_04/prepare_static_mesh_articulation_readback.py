"""Publish a fresh inspector readback with the existing cabinet Mesh as fixed owner.

This bounded inspector never authors USD or joints. Native preparation publication
still consumes and validates this draft. Existing output files are never replaced.
"""
import argparse
import hashlib
import json
from pathlib import Path

CABINET = '/Asset/drawer_cabinet_0'
STATIC_MESH = CABINET + '/Primitive_0'
DRAWER = '/Asset/drawer_cabinet_drawer_01_1'
IMPL = 'prepare_static_mesh_articulation_readback.py'
OUTPUTS = ('inspector_config.json', 'saved_geometry_observation.json', IMPL,
           'articulation_preparation_readback.json')


def membership_rows(hierarchy, mesh_paths):
    """Only the cabinet parent and its sole Mesh change authoritative ownership."""
    if len(mesh_paths) != 5 or len(set(mesh_paths)) != 5 or STATIC_MESH not in mesh_paths:
        raise ValueError('Exact five-mesh drawer assembly required')
    parents = {path.rsplit('/', 1)[0] for path in mesh_paths}
    if len(parents) != 5 or DRAWER not in parents:
        raise ValueError('Expected five distinct part parents and original upper drawer')
    owners = (parents - {CABINET}) | {STATIC_MESH}
    paths = [row['prim_path'] for row in hierarchy]
    if len(set(paths)) != len(paths) or not {'/Asset', CABINET, STATIC_MESH, DRAWER}.issubset(paths):
        raise ValueError('Missing or duplicate authoritative hierarchy paths')
    rows = []
    for path in paths:
        candidates = [owner for owner in owners if path == owner or path.startswith(owner + '/')]
        owner = max(candidates, key=len) if candidates else '/Asset'
        disposition = ('independent_motion' if path == DRAWER else
                       'co_rigid' if owner == DRAWER else 'explicit_fixed')
        rows.append({'member_prim': path, 'authoritative_owner_prim': owner,
                     'disposition': disposition})
    return rows


def main():
    from pxr import Ar, Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils
    from content_agent_workflows.articulation.preparation import (
        ArticulationPreparationInspectionReadbackDraft,
        ArticulationPreparationInspectorConfiguration,
        _authoritative_memberships,
    )
    from joint_agent.functions.articulation_contract_v2_frames import (
        _require_endpoint, _require_invertible_world_transform,
        _require_static_endpoint_transform,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--saved', default='saved_inspection.usda')
    parser.add_argument('--render', action='append', required=True)
    parser.add_argument('--scene', action='append', required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    for name in OUTPUTS:
        if (root / name).exists() or (root / name).is_symlink():
            raise FileExistsError(f'Refusing to replace an existing preparation artifact: {name}')

    def safe_path(relative):
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('All inputs must use root-contained relative paths')
        path = root / relative
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f'Not a regular contained input: {relative}')
        if any(parent.is_symlink() for parent in path.parents if parent != root and root in parent.parents):
            raise ValueError(f'Symlink ancestor in input: {relative}')
        return path

    def byte_claim(name, data):
        return {'relative_path': name, 'sha256': hashlib.sha256(data).hexdigest(),
                'size_bytes': len(data)}

    def claim(name):
        return byte_claim(Path(name).as_posix(), safe_path(name).read_bytes())

    original_claims = {name: claim(name) for name in {args.source, args.saved, *args.render, *args.scene}}
    stage = Usd.Stage.Open(str(safe_path(args.saved)))
    source = Usd.Stage.Open(str(safe_path(args.source)))
    if not stage or not source or UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError('Readable meter-scale source and saved stage required')
    for inspected in (source, stage):
        if any(prim.IsA(UsdPhysics.Joint) for prim in inspected.TraverseAll()):
            raise ValueError('Fresh joint-free preparation required')
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    mesh_paths = [str(prim.GetPath()) for prim in meshes]
    if sorted(mesh_paths) != sorted(str(prim.GetPath()) for prim in source.Traverse() if prim.IsA(UsdGeom.Mesh)):
        raise ValueError('Source and saved mesh hierarchy must agree')
    cache = UsdGeom.XformCache()
    for path in (STATIC_MESH, DRAWER):
        prim = _require_endpoint(stage, path, label='static-Mesh inspector', Sdf=Sdf, UsdGeom=UsdGeom)
        _require_static_endpoint_transform(prim, label='static-Mesh inspector')
        _require_invertible_world_transform(stage, prim, label='static-Mesh inspector')
    for path in (CABINET, STATIC_MESH, DRAWER):
        prim = stage.GetPrimAtPath(path)
        if not prim or cache.GetLocalToWorldTransform(prim) != Gf.Matrix4d(1):
            raise ValueError(f'Bounded route requires identical identity world frames: {path}')
    cabinet = stage.GetPrimAtPath(CABINET)
    if [str(child.GetPath()) for child in cabinet.GetChildren() if child.IsA(UsdGeom.Mesh)] != [STATIC_MESH]:
        raise ValueError('Cabinet must have the unique original child Mesh')
    hierarchy = [{'prim_path': str(prim.GetPath()),
                  'parent_prim_path': str(prim.GetPath()).rsplit('/', 1)[0] or None,
                  'type_name': prim.GetTypeName(), 'active': bool(prim.IsActive())}
                 for prim in stage.Traverse()]
    memberships = membership_rows(hierarchy, mesh_paths)
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    owner_by_path = {row['member_prim']: row['authoritative_owner_prim'] for row in memberships}
    observations = []
    for prim in meshes:
        path = str(prim.GetPath())
        original = source.GetPrimAtPath(path)
        for attribute in ('points', 'faceVertexCounts', 'faceVertexIndices'):
            if prim.GetAttribute(attribute).Get() != original.GetAttribute(attribute).Get():
                raise ValueError(f'Saved source geometry differs: {path}.{attribute}')
        if cache.GetLocalToWorldTransform(prim) != UsdGeom.XformCache().GetLocalToWorldTransform(original):
            raise ValueError(f'Saved source world frame differs: {path}')
        bounds = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        mesh = UsdGeom.Mesh(prim)
        observations.append({'path': path, 'owner': owner_by_path[path],
                             'world_bounds_m': [list(bounds.GetMin()), list(bounds.GetMax())],
                             'vertices': len(mesh.GetPointsAttr().Get()),
                             'faces': len(mesh.GetFaceVertexCountsAttr().Get())})
    configuration = ArticulationPreparationInspectorConfiguration.model_validate({
        'membership_policy': 'retained-explicit-membership-v1', 'memberships': memberships,
        'capabilities': {'canonical_output_evidence_required': True},
    })
    config_bytes = (configuration.model_dump_json(indent=2) + '\n').encode()
    observation_bytes = (json.dumps({
        'source': args.source, 'saved_stage': args.saved,
        'up_axis': str(UsdGeom.GetStageUpAxis(stage)), 'meters_per_unit': 1,
        'moving_drawer': DRAWER, 'static_cabinet_owner': STATIC_MESH,
        'cabinet_parent_owner': '/Asset', 'identical_identity_endpoint_world_frames': True,
        'parts': observations,
        'scope': 'Explicit source-bound static membership only; no joint or physical success claim.',
    }, indent=2) + '\n').encode()
    dependencies = set()
    for filename in {args.source, args.saved}:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(safe_path(filename)))
        if unresolved:
            raise ValueError(f'Unresolved dependencies: {unresolved}')
        for locator in [layer.realPath for layer in layers] + list(assets):
            outer = Ar.SplitPackageRelativePathOuter(locator)[0] if Ar.IsPackageRelativePath(locator) else locator
            path = Path(outer).resolve(strict=True)
            if path != safe_path(filename).resolve():
                dependencies.add(path.relative_to(root).as_posix())
    dependency_claims = [claim(name) for name in sorted(dependencies)]
    original_claims.update({item['relative_path']: item for item in dependency_claims})
    implementation_bytes = Path(__file__).read_bytes()
    draft = ArticulationPreparationInspectionReadbackDraft.model_validate({
        'inspector_id': 'capstone-static-mesh-source-usdcli-readback',
        'inspector_implementation': IMPL + '.v1',
        'source': original_claims[args.source], 'saved_stage': original_claims[args.saved],
        'dependencies': dependency_claims, 'dependency_entry_count': len(dependency_claims),
        'configuration': byte_claim('inspector_config.json', config_bytes),
        'inspector_implementation_artifact': byte_claim(IMPL, implementation_bytes),
        'hierarchy': hierarchy, 'memberships': memberships,
        'render_artifacts': [original_claims[name] for name in args.render],
        'scene_artifacts': [original_claims[name] for name in args.scene] +
                           [byte_claim('saved_geometry_observation.json', observation_bytes)],
        'proposal_status': 'not_requested',
    })
    _authoritative_memberships(draft, configuration)
    if any(claim(name) != expected for name, expected in original_claims.items()):
        raise ValueError('An input changed during inspection')
    outputs = {'inspector_config.json': config_bytes,
               'saved_geometry_observation.json': observation_bytes,
               IMPL: implementation_bytes,
               'articulation_preparation_readback.json': (draft.model_dump_json(indent=2) + '\n').encode()}
    for name, data in outputs.items():
        with (root / name).open('xb') as stream:
            stream.write(data)
    print(json.dumps({'readback': str(root / 'articulation_preparation_readback.json'),
                      'readback_sha256': hashlib.sha256(outputs['articulation_preparation_readback.json']).hexdigest(),
                      'static_owner': STATIC_MESH, 'moving_drawer': DRAWER,
                      'prims': len(hierarchy), 'dependencies': sorted(dependencies),
                      'all_inputs_unchanged': all(claim(name) == expected for name, expected in original_claims.items())}, indent=2))


if __name__ == '__main__':
    main()
