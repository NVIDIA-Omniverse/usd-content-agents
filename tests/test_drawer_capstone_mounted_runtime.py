# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic placement and acceptance witnesses; no native solver is launched."""
import json
from pathlib import Path

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from physics_agent.tuning.scenarios._scene_builder import build_drop_settle_scene
from content_agent_workflows.physics import scene_ops, usd_cli_ops, workflow
from content_workflow_cli import cli, runner


def fixture(path, *, world_anchor=False, meter_scale=1.0, separate_roots=False):
    stage=Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage,meter_scale)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.y)
    root=UsdGeom.Xform.Define(stage,'/Asset');stage.SetDefaultPrim(root.GetPrim())
    root.AddTranslateOp().Set(Gf.Vec3d(0.2,0.4,0.6))
    static=UsdGeom.Xform.Define(stage,'/Asset/Static')
    static.AddTranslateOp().Set(Gf.Vec3d(0.3,0.8,0.1))
    paths=['/Asset/Drawer']
    if separate_roots:
        UsdGeom.Xform.Define(stage,'/Other');paths.append('/Other/Drawer')
    for i,path_name in enumerate(paths):
        body=UsdGeom.Xform.Define(stage,path_name)
        body.AddTranslateOp().Set(Gf.Vec3d(0.3,0.8,0.1))
        rb=UsdPhysics.RigidBodyAPI.Apply(body.GetPrim());rb.CreateKinematicEnabledAttr(False)
        mass=UsdPhysics.MassAPI.Apply(body.GetPrim());mass.CreateMassAttr(2.0)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.1,.1,.1))
        cube=UsdGeom.Cube.Define(stage,path_name+'/Collider');cube.CreateSizeAttr(.2)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        joint=UsdPhysics.PrismaticJoint.Define(stage,f'/Asset/Joints/J{i}')
        joint.CreateAxisAttr('Z');joint.CreateLowerLimitAttr(0);joint.CreateUpperLimitAttr(.3)
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
        if not world_anchor and not separate_roots:
            joint.CreateBody0Rel().SetTargets([static.GetPath()])
            joint.CreateLocalPos0Attr(Gf.Vec3f(0));joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        else:
            p=body.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
            joint.CreateLocalPos0Attr(Gf.Vec3f(p));joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1));joint.CreateLocalRot1Attr(Gf.Quatf(1))
    stage.GetRootLayer().Save()
    return paths


def pose(stage,path):
    return np.array(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))


def joint_frames(stage):
    rows=[]
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):continue
        joint=UsdPhysics.Joint(prim);row=[]
        for i in (0,1):
            targets=getattr(joint,f'GetBody{i}Rel')().GetTargets()
            M=UsdGeom.Xformable(stage.GetPrimAtPath(targets[0])).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) if targets else Gf.Matrix4d(1)
            row.append(list(M.Transform(Gf.Vec3d(getattr(joint,f'GetLocalPos{i}Attr')().Get()))))
        rows.append(row)
    return np.array(rows)


@pytest.mark.parametrize('world_anchor',[False,True])
@pytest.mark.parametrize('drop_height',[None,0.0])
def test_mounted_preserves_static_and_world_joint_frames(tmp_path,world_anchor,drop_height):
    source=tmp_path/'mounted.usda';paths=fixture(source,world_anchor=world_anchor)
    before=source.read_bytes();original=Usd.Stage.Open(str(source))
    output=tmp_path/'scene.usda'
    info=build_drop_settle_scene(source,output,placement_mode='mounted',drop_height_m=drop_height)
    saved=Usd.Stage.Open(str(output))
    for path in ['/Asset','/Asset/Static',*paths]:
        np.testing.assert_array_equal(pose(original,path),pose(saved,path))
    np.testing.assert_array_equal(joint_frames(original),joint_frames(saved))
    np.testing.assert_allclose(joint_frames(saved)[:,0],joint_frames(saved)[:,1],atol=1e-7)
    assert info['rest_position']==pytest.approx(pose(original,paths[0])[3,:3])
    assert info['placement_mode']=='mounted' and info['drop_height_m_resolved']==0
    assert source.read_bytes()==before
    scenes=[UsdPhysics.Scene(p) for p in saved.Traverse() if p.IsA(UsdPhysics.Scene)]
    assert scenes and float(scenes[0].GetGravityMagnitudeAttr().Get())==pytest.approx(9.81)
    assert tuple(scenes[0].GetGravityDirectionAttr().Get())==(0,-1,0)
    assert saved.GetPrimAtPath('/Asset/Joints/J0').GetAttribute('physics:upperLimit').Get()==pytest.approx(.3)


def test_default_drop_still_places_body_at_ground(tmp_path):
    source=tmp_path/'drop.usda';paths=fixture(source)
    output=tmp_path/'scene.usda';info=build_drop_settle_scene(source,output,drop_height_m=0)
    stage=Usd.Stage.Open(str(output))
    bound=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_]).ComputeWorldBound(stage.GetPrimAtPath(paths[0])).ComputeAlignedRange()
    assert bound.GetMin()[1]==pytest.approx(0,abs=1e-7)
    assert info['placement_mode']=='drop'
    assert np.linalg.norm(joint_frames(stage)[0,0]-joint_frames(stage)[0,1])>.5


@pytest.mark.parametrize('kwargs,match',[
    ({'placement_mode':'unknown'},'placement_mode'),
    ({'placement_mode':'mounted','drop_height_m':.1},'drop_height'),
    ({'placement_mode':'mounted','drop_height_m':-1},'drop_height'),
])
def test_invalid_mounted_request_is_rejected_before_export(tmp_path,kwargs,match):
    source=tmp_path/'input.usda';fixture(source);before=source.read_bytes();output=tmp_path/'scene.usda'
    with pytest.raises(ValueError,match=match):build_drop_settle_scene(source,output,**kwargs)
    assert not output.exists() and source.read_bytes()==before


def test_mounted_nonmetric_fails_without_rebasing_joint_or_geometry(tmp_path):
    source=tmp_path/'centimeter.usda';fixture(source,meter_scale=.01);before=source.read_bytes()
    with pytest.raises(ValueError,match='metersPerUnit=1'):
        build_drop_settle_scene(source,tmp_path/'scene.usda',placement_mode='mounted')
    assert source.read_bytes()==before


@pytest.mark.parametrize('fault',[None,'jump','nonfinite','unsettled'])
def test_mounted_runtime_keeps_continuity_finite_and_settling_gates(tmp_path,monkeypatch,fault):
    source=tmp_path/'input.usda';paths=fixture(source,world_anchor=True)
    def recorded_solver(**kwargs):
        assert kwargs['body_path']==paths[0]
        stage=Usd.Stage.Open(str(kwargs['scene_usd']))
        np.testing.assert_allclose(joint_frames(stage)[:,0],joint_frames(stage)[:,1],atol=1e-7)
        initial=list(kwargs['rest_position']);out=Path(kwargs['output_dir']);out.mkdir(parents=True)
        rows=[]
        for i in range(91):
            p=initial.copy();v=[0.0]*6
            if fault=='jump' and i>0:p[0]+=3.0
            if fault=='nonfinite' and i==1:p[0]=float('nan')
            if fault=='unsettled':p[2]+=.01*i;v[2]=.3
            rows.append({'t':i/30,'pose':p+[0,0,0,1],'vel':v})
        trajectory=out/'trajectory.jsonl';trajectory.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        recording=out/'recording.usda';recording.write_text('#usda 1.0\n')
        return {'n_bodies':1,'trajectory_jsonl':str(trajectory),'recording_usda':str(recording),'simulation_facts':{'reported_step_count':720}}
    monkeypatch.setattr(usd_cli_ops,'simulate_physics_scene',recorded_solver)
    result=scene_ops.validate_runtime(physics_usd=source,output_dir=tmp_path/'runtime',placement_mode='mounted')
    assert result['acceptance']['require_settle'] is True
    assert result['acceptance']['detect_initial_pose_discontinuity'] is True
    assert result['acceptance']['require_gravity_response'] is False
    if fault=='jump':assert any('discontinuous' in x for x in result['failures'])
    elif fault=='nonfinite':assert any('non-finite' in x for x in result['failures'])
    elif fault=='unsettled':assert any('settle threshold' in x for x in result['warnings'])
    else:assert result['failures']==[] and result['summary']['settle_time_s'] is not None


def test_mounted_cannot_disable_settling(tmp_path):
    with pytest.raises(ValueError,match='require_settle'):
        scene_ops.validate_runtime(physics_usd=tmp_path/'unused.usda',output_dir=tmp_path/'runtime',placement_mode='mounted',acceptance={'require_settle':False})


def test_cli_and_recorded_request_preserve_explicit_mode(tmp_path):
    source=tmp_path/'input.usda';fixture(source)
    args=cli.build_parser().parse_args(['physics','apply','--usd',str(source),'--output-dir',str(tmp_path/'run'),'--runtime-placement-mode','mounted'])
    assert args.runtime_placement_mode=='mounted'
    config=runner.PhysicsApplyConfig(repo_root=Path(__file__).resolve().parents[1],usd_path=source,runtime_placement_mode='mounted')
    assert runner._build_physics_request(config,tmp_path/'run')['runtime_validation']['runtime_placement_mode']=='mounted'
    assert runner._physics_run_manifest_policy(config)['runtime_placement_mode']=='mounted'
    params=workflow.PhysicsApplyWorkflowInput(usd_path=source,output_dir=tmp_path/'run',runtime_placement_mode='mounted')
    assert params.model_dump()['runtime_placement_mode']=='mounted'
    assert runner.PhysicsApplyConfig(repo_root=tmp_path,usd_path=source).runtime_placement_mode=='drop'


def test_single_body_workflow_forwards_mounted_mode(tmp_path,monkeypatch):
    captured={}
    def validate(**kwargs):
        captured.update(kwargs)
        return {'engine':'none','not_evaluated':True,'failures':[],'warnings':[]}
    monkeypatch.setattr(scene_ops,'validate_runtime',validate)
    workflow.validate_physics_runtime(physics_usd=tmp_path/'unused.usda',output_dir=tmp_path/'runtime',placement_mode='mounted')
    assert captured['placement_mode']=='mounted'


def test_multibody_mounted_preserves_world_anchors_without_common_root(tmp_path):
    source=tmp_path/'separate.usda';paths=fixture(source,world_anchor=True,separate_roots=True)
    original=Usd.Stage.Open(str(source));frames=joint_frames(original)
    _evidence,report_path=workflow.validate_physics_runtime_multi_body(
        physics_usd=source,output_dir=tmp_path/'runtime',body_prim_paths=paths,
        engine='fake',placement_mode='mounted',duration_s=.1)
    report=json.loads(report_path.read_text())
    assert report['placement_prim_path'] is None and report['placement_mode']=='mounted'
    assert len(report['per_body_results'])==2
    for result in report['per_body_results']:
        stage=Usd.Stage.Open(result['scene_usd'])
        np.testing.assert_array_equal(frames,joint_frames(stage))
        for path in paths:np.testing.assert_array_equal(pose(original,path),pose(stage,path))


def test_tuned_candidate_revalidation_forwards_mounted_mode(tmp_path,monkeypatch):
    from content_agent_workflows import physics
    captured={}
    def validate(**kwargs):
        captured.update(kwargs)
        raise RuntimeError('Synthetic stop before any solver')
    monkeypatch.setattr(physics,'validate_physics_runtime',validate)
    class Trace:
        def write(self,*args,**kwargs):pass
    config=runner.PhysicsApplyConfig(repo_root=tmp_path,usd_path=tmp_path/'source.usda',runtime_placement_mode='mounted')
    runner._revalidate_tuned_physics_usd(config=config,run_dir=tmp_path,tuned_usd=tmp_path/'candidate.usda',trace_writer=Trace())
    assert captured['placement_mode']=='mounted'
