"""Synthetic authoring/regression tests; no scored source or solver runs."""
import os
from pathlib import Path

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics
from content_agent_workflows.physics.workflow import (
    PhysicsComponent, PhysicsComponentDecision, PhysicsComponentTargetDecision,
    PhysicsConvexDecompositionOptions, _merge_rebased_component_decisions,
    add_physics_component_target_catalog, resolve_physics_v2_patch_targets,
)
from content_agent_workflows.physics.usd_cli_ops import physics_patch_from_workflow_decisions
from usd_core.physics import apply_collision, apply_operations

OPTIONS = dict(shrink_wrap=True, error_percentage=0.5, hull_vertex_limit=128,
               max_convex_hulls=64, voxel_resolution=1_000_000)
NATIVE = dict(shrink_wrap=('shrinkWrap', Sdf.ValueTypeNames.Bool, False),
              error_percentage=('errorPercentage', Sdf.ValueTypeNames.Float, 10.0),
              hull_vertex_limit=('hullVertexLimit', Sdf.ValueTypeNames.Int, 64),
              max_convex_hulls=('maxConvexHulls', Sdf.ValueTypeNames.Int, 32),
              voxel_resolution=('voxelResolution', Sdf.ValueTypeNames.Int, 500000))
PREFIX = 'physxConvexDecompositionCollision:'


def decision_patch(options=OPTIONS):
    component = PhysicsComponent(component_id='body', body_root_path='/World/Body',
                                 visual_evidence_paths=['/World/Body/Mesh'])
    catalog = add_physics_component_target_catalog({'components': [component.model_dump()]})
    decision = dict(decision_id='body', component_id='body',
                    collider_target_ids=[catalog['components'][0]['authoring_targets'][0]['target_id']],
                    collision_mode='author_on_targets', inferred_material_family='wood',
                    collision_approximation='convexDecomposition',
                    physical_properties={'estimated_mass_kg':1.0,'density':500.0},
                    confidence=1.0, rationale='Synthetic concave cooking fixture.')
    if options is not None:
        decision['convex_decomposition'] = options
    PhysicsComponentTargetDecision.model_validate(decision)
    payload = dict(schema_version='content-agent-workflows.physics-decision-patch.v2',
                   source_digest='sha256:fixture', decisions=[decision], unresolved_components=[])
    resolved = resolve_physics_v2_patch_targets(payload, components=[component], source_digest='sha256:fixture')[1]
    return component, resolved


def stage_with_mesh():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World/Body')
    UsdGeom.Mesh.Define(stage, '/World/Body/Mesh')
    return stage


def test_target_resolution_native_authoring_and_saved_readback(tmp_path):
    _, decisions = decision_patch()
    assert decisions[0].convex_decomposition.model_dump() == OPTIONS
    operations = physics_patch_from_workflow_decisions(
        [d.model_dump(mode='json') for d in decisions], author_rigid_body=True,
        physics_scene_path='/World/PhysicsScene')
    assert operations['colliders'][0]['convex_decomposition'] == OPTIONS
    stage = stage_with_mesh()
    authored = apply_operations(stage, operations)
    assert 'PhysxConvexDecompositionCollisionAPI' in authored['authored_apis']['/World/Body/Mesh']
    path = tmp_path/'fixture.usda';stage.GetRootLayer().Export(str(path))
    saved = Usd.Stage.Open(str(path));prim=saved.GetPrimAtPath('/World/Body/Mesh')
    assert 'PhysxConvexDecompositionCollisionAPI' in prim.GetMetadata('apiSchemas').GetAppliedItems()
    for key,(suffix,kind,_) in NATIVE.items():
        attr=prim.GetAttribute(PREFIX+suffix)
        assert attr.GetTypeName()==kind and attr.Get()==OPTIONS[key] and not attr.IsCustom()


def test_omitted_record_preserves_legacy_usd_and_operations():
    _, decisions = decision_patch(None)
    operations = physics_patch_from_workflow_decisions(
        [d.model_dump(mode='json') for d in decisions], author_rigid_body=True,
        physics_scene_path='/World/PhysicsScene')
    assert operations['colliders']==[{'path':'/World/Body/Mesh','approximation':'convexDecomposition'}]
    stage=stage_with_mesh();apply_collision(stage,'/World/Body/Mesh',approximation='convexDecomposition')
    expected=stage.GetRootLayer().ExportToString()
    apply_collision(stage,'/World/Body/Mesh',approximation='convexDecomposition',convex_decomposition=None)
    assert stage.GetRootLayer().ExportToString()==expected
    assert not any(a.GetName().startswith(PREFIX) for a in stage.GetPrimAtPath('/World/Body/Mesh').GetAuthoredAttributes())


def test_partial_record_authors_documented_defaults_without_changing_other_cooking_fields():
    stage=stage_with_mesh();prim=stage.GetPrimAtPath('/World/Body/Mesh')
    prim.CreateAttribute(PREFIX+'minThickness',Sdf.ValueTypeNames.Float,custom=False).Set(.004)
    expected=PhysicsConvexDecompositionOptions(shrink_wrap=True).model_dump()
    apply_collision(stage,'/World/Body/Mesh',approximation='convexDecomposition',
                    convex_decomposition={'shrink_wrap':True})
    for key,(suffix,_,_) in NATIVE.items():
        assert prim.GetAttribute(PREFIX+suffix).Get()==expected[key]
    assert prim.GetAttribute(PREFIX+'minThickness').Get()==pytest.approx(.004)


def test_omitted_record_preserves_existing_custom_cooking_values_byte_for_byte():
    stage=stage_with_mesh()
    apply_collision(stage,'/World/Body/Mesh',approximation='convexDecomposition',
                    convex_decomposition=OPTIONS)
    before=stage.GetRootLayer().ExportToString()
    apply_collision(stage,'/World/Body/Mesh',approximation='convexDecomposition')
    assert stage.GetRootLayer().ExportToString()==before


@pytest.mark.parametrize('key,value', [
    ('shrink_wrap',1),('shrink_wrap','true'),('shrink_wrap',None),
    ('error_percentage',True),('error_percentage','1'),('error_percentage',float('nan')),
    ('error_percentage',float('inf')),('error_percentage',-0.1),('error_percentage',100.1),
    ('hull_vertex_limit',3),('hull_vertex_limit',256),('hull_vertex_limit',64.0),
    ('max_convex_hulls',0),('max_convex_hulls',257),('max_convex_hulls',True),
    ('voxel_resolution',9999),('voxel_resolution',4000001),('voxel_resolution','500000'),
    ('unexpected',1),
])
def test_invalid_record_rejected_before_all_batch_mutations(key,value):
    bad={**OPTIONS,key:value}
    with pytest.raises(ValueError):PhysicsConvexDecompositionOptions.model_validate(bad)
    stage=stage_with_mesh();before=stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError):
        apply_operations(stage,{'scene_paths':['/World/PhysicsScene'],
            'rigid_bodies':[{'path':'/World/Body','mass':1}],
            'colliders':[{'path':'/World/Body/Mesh','approximation':'convexHull'},
                         {'path':'/World/Body/Mesh','approximation':'convexDecomposition','convex_decomposition':bad}]})
    assert stage.GetRootLayer().ExportToString()==before


@pytest.mark.parametrize('kind,approximation',[('Mesh','convexHull'),('Cube','convexDecomposition')])
def test_incompatible_target_rejected_without_mutation(kind,approximation):
    stage=stage_with_mesh()
    if kind=='Cube':stage.GetPrimAtPath('/World/Body/Mesh').SetTypeName('Cube')
    before=stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError,match='requires a Mesh'):
        apply_operations(stage,{'scene_paths':['/World/PhysicsScene'],
            'colliders':[{'path':'/World/Body/Mesh','approximation':approximation,'convex_decomposition':OPTIONS}]})
    assert stage.GetRootLayer().ExportToString()==before


def test_typed_decision_rejects_wrong_approximation():
    _,decisions=decision_patch()
    with pytest.raises(ValueError,match='requires convexDecomposition'):
        PhysicsComponentDecision.model_validate({**decisions[0].model_dump(), 'collision_approximation':'none'})


def test_existing_wrong_attribute_type_rejected_before_any_mutation():
    stage=stage_with_mesh();prim=stage.GetPrimAtPath('/World/Body/Mesh')
    prim.CreateAttribute(PREFIX+'hullVertexLimit',Sdf.ValueTypeNames.String).Set('64')
    before=stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError,match='incompatible existing'):
        apply_operations(stage,{'scene_paths':['/World/PhysicsScene'],
            'colliders':[{'path':'/World/Body/Mesh','approximation':'convexDecomposition','convex_decomposition':OPTIONS}]})
    assert stage.GetRootLayer().ExportToString()==before


def test_rebase_rejects_conflicting_options_and_preserves_equal_options():
    component,decisions=decision_patch();first=decisions[0]
    second=first.model_copy(update={'decision_id':'second'})
    assert _merge_rebased_component_decisions(component,[first,second]).convex_decomposition==first.convex_decomposition
    conflict=second.model_copy(update={'convex_decomposition':PhysicsConvexDecompositionOptions()})
    with pytest.raises(RuntimeError,match='incompatible convex_decomposition'):
        _merge_rebased_component_decisions(component,[first,conflict])
    omitted=second.model_copy(update={'convex_decomposition':None})
    with pytest.raises(RuntimeError,match='incompatible convex_decomposition'):
        _merge_rebased_component_decisions(component,[first,omitted])


def test_installed_native_schema_names_types_defaults():
    path=os.environ.get('OVPHYSX_GENERATED_SCHEMA')
    if not path:pytest.skip('Set actual isolated native schema path for runtime qualification')
    layer=Sdf.Layer.FindOrOpen(str(Path(path)))
    defaults=PhysicsConvexDecompositionOptions().model_dump()
    for key,(suffix,kind,default) in NATIVE.items():
        spec=layer.GetAttributeAtPath('/PhysxConvexDecompositionCollisionAPI.'+PREFIX+suffix)
        assert spec and spec.typeName==kind and spec.default==default==defaults[key]
