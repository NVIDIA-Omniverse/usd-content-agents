"""Analytical trace mutations and retained USD-copy controls; no simulation."""
import copy
import json
import math
import os
from pathlib import Path
import numpy as np
import pytest
import audit as A

ROOT=Path(os.environ.get('TASK_AUDIT_ROOT','/opt/astra-content-value-20260921'))
CAP=ROOT/'capstone'
SPEC=A.read(CAP/'evaluator/source_clear_v1/drawer_acceptance.json')

@pytest.fixture
def static_trace():
    # Deliberately stationary mathematical trace, not a claimed solver trajectory.
    rows=[];rng=np.random.default_rng(11)
    for i in range(2460):
        t=i*SPEC['dt_s'];phase,target,speed=A.phase_target(t,SPEC);force=0.
        if t>=SPEC['phase_durations_s']['settle']:
            c=SPEC['controller'];force=np.clip(c['kp_n_m']*target+c['kd_ns_m']*speed+rng.uniform(-c['force_jitter_n'],c['force_jitter_n']),-c['force_limit_n'],c['force_limit_n'])
        rows.append({'step':i,'time_s':(i+1)*SPEC['dt_s'],'phase':phase,'target_q_m':target,'q_m':0.,'drawer_pose':[0.,0.,0.,0.,0.,0.,1.], 'payload_pose':[0.,1.05,0.,0.,0.,0.,1.], 'drawer_velocity':[0.]*6,'payload_velocity':[0.]*6,'applied_force_world_n':[0.,0.,float(np.float32(force))],'payload_drawer_contact_force_n':[[[0.,0.02,0.]]],'contacts':[{'p':[0.,1.,0.],'normal':[0.,1.,0.],'impulse':[0.,0.02,0.],'separation':-.002}]})
    request={'spec':SPEC,'seed':11,'initial_drawer_pose':[0.,0.,0.,0.,0.,0.,1.]}
    names=['max_q_m','max_penetration_m','max_off_axis_m','max_rotation_deg','max_linear_speed_m_s','max_angular_speed_rad_s','payload_contact_samples','payload_retained','max_force_n','initial_drift_m','min_open_hold_q_m','max_closed_hold_error_m','max_closed_hold_speed_m_s']
    checks=['opening','closing','closed_settled','payload_retained','payload_contact','penetration','off_axis','rotation','initial_drift','force_bounded','finite_and_stable']
    report={'import_pose_error_m':0.,'metrics':{n:0. for n in names},'checks':{n:True for n in checks}}
    return rows,request,report


def compute(fixture):return A.recompute(*fixture)


def test_analytical_stationary_trace_is_failed_opening(static_trace):
    r=compute(static_trace)
    assert r['metrics']['min_open_hold_q_m']==0
    assert r['metrics']['max_closed_hold_error_m']==0
    assert r['metrics']['max_closed_hold_speed_m_s']==0
    assert r['metrics']['payload_contact_samples']==2280
    assert r['metrics']['max_penetration_m']==.002
    assert r['independently_corroborated_failures']==['opening']
    assert not r['initial_native_xy_and_quaternion_recorded']

@pytest.mark.parametrize('mutation,check',[('settling','closed_settled'),('payload_escape','payload_retained'),('no_contact','payload_contact'),('penetration','penetration'),('off_axis','off_axis'),('orientation','rotation')])
def test_physical_trace_mutation_detected(static_trace,mutation,check):
    rows,req,report=static_trace
    if mutation=='settling':rows[-1]['drawer_velocity'][0]=.1
    elif mutation=='payload_escape':rows[-1]['payload_pose'][0]=.60
    elif mutation=='no_contact':
        for r in rows:r['payload_drawer_contact_force_n']=[[[0.,0.,0.]]]
    elif mutation=='penetration':rows[-1]['contacts'][0]['separation']=-.006
    elif mutation=='off_axis':rows[-1]['drawer_pose'][0]=.011
    elif mutation=='orientation':rows[-1]['drawer_pose'][3:]=[math.sin(math.radians(3)/2),0.,0.,math.cos(math.radians(3)/2)]
    assert compute(static_trace)['trace_checks'][check] is False


def test_wrong_force_controller_rejected(static_trace):
    static_trace[0][-1]['applied_force_world_n'][2]=41.
    with pytest.raises(ValueError,match='seeded bounded PD'):compute(static_trace)


def test_wrong_phase_rejected(static_trace):
    static_trace[0][179]['phase']='open'
    with pytest.raises(ValueError,match='Controller phase'):compute(static_trace)


def test_wrong_q_origin_rejected(static_trace):
    static_trace[0][-1]['q_m']=.01
    with pytest.raises(ValueError,match='native Z origin'):compute(static_trace)

@pytest.mark.parametrize('kind',['missing','duplicate','nonfinite','wrong_time'])
def test_trace_schema_fail_closed(static_trace,tmp_path,kind):
    rows=static_trace[0]
    if kind=='missing':rows.pop()
    elif kind=='duplicate':rows[-1]['step']=0
    elif kind=='nonfinite':rows[-1]['drawer_pose'][0]=float('nan')
    else:rows[-1]['time_s']+=SPEC['dt_s']
    path=tmp_path/'trace.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):A.trace_rows(path,SPEC)


def test_post_step_phase_boundary_and_impulse_units(static_trace):
    rows=static_trace[0]
    assert rows[179]['time_s']==.75 and rows[179]['phase']=='settle'
    assert rows[180]['phase']=='open'
    r=compute(static_trace)
    assert r['contact_detector']['threshold_raw_impulse_ns']==.01
    assert r['contact_detector']['threshold_equivalent_force_n']==2.4


def test_actual_retained_scene_settings_match():
    trial=CAP/'evaluations/native09_source_clear_v1/seed_11'
    r=A.source_scene_check(CAP/'runs/drawer_physics_09/physics.usda',trial/'scene.usda',A.read(trial/'request.json'),A.read(CAP/'evidence/native09_task_bindings.json'))
    assert r['only_declared_scene_interventions']

@pytest.mark.parametrize('kind',['payload_mass','source_mass'])
def test_copied_scene_property_mutation_rejected(tmp_path,kind):
    from pxr import Usd
    trial=CAP/'evaluations/native09_source_clear_v1/seed_11';path=tmp_path/'mutated.usda';path.write_bytes((trial/'scene.usda').read_bytes())
    stage=Usd.Stage.Open(str(path));body='/__EvaluatorPayload' if kind=='payload_mass' else '/Asset/drawer_cabinet_drawer_01_1'
    stage.GetPrimAtPath(body).GetAttribute('physics:mass').Set(12.);stage.GetRootLayer().Save()
    with pytest.raises(ValueError):A.source_scene_check(CAP/'runs/drawer_physics_09/physics.usda',path,A.read(trial/'request.json'),A.read(CAP/'evidence/native09_task_bindings.json'))


def test_reported_metric_and_verdict_mutation_detected(static_trace):
    rows,request,report=static_trace
    initial=compute(static_trace)
    report['metrics']=copy.deepcopy(initial['metrics'])
    report['checks']=copy.deepcopy(initial['trace_checks'])
    assert compute(static_trace)['all_reported_metrics_consistent']
    assert compute(static_trace)['reported_trace_checks_match']
    report['metrics']['max_closed_hold_error_m']=.1
    report['checks']['opening']=True
    changed=compute(static_trace)
    assert not changed['all_reported_metrics_consistent']
    assert not changed['reported_trace_checks_match']


def test_force_serialization_tolerance_does_not_relax_physical_bound(static_trace):
    rows,request,report=static_trace
    request['spec']=copy.deepcopy(SPEC);request['spec']['controller']['kp_n_m']=1000.
    c=request['spec']['controller'];rng=np.random.default_rng(11)
    for r in rows:
        t=r['step']*SPEC['dt_s'];phase,target,target_v=A.phase_target(t,request['spec'])
        force=0. if t<SPEC['phase_durations_s']['settle'] else float(np.clip(c['kp_n_m']*target+c['kd_ns_m']*target_v+rng.uniform(-c['force_jitter_n'],c['force_jitter_n']),-40,40))
        r['applied_force_world_n'][2]=float(np.float32(force))
    assert rows[1230]['applied_force_world_n'][2]==40.
    rows[1230]['applied_force_world_n'][2]=float(np.nextafter(np.float32(40),np.float32(float('inf'))))
    r=compute(static_trace)
    assert r['max_force_trace_float32_comparison_error_n']<A.FORCE_ATOL
    assert r['trace_checks']['force_bounded'] is False
