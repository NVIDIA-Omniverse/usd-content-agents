"""Read-only independent drawer trace audit. Never imports or runs the solver.

Audit consistency is distinct from task acceptance. Missing initial native XY/
quaternion and actor-less global contact records impose explicit evidence limits.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

FROZEN = {
    'drawer_acceptance.json': '90cde935ce539607c6a770f6ff0ee8cc8c5bc37ba2f18c76802e5cdfe70bf1cd',
    'drawer_evaluate.py': 'dcd7f4e4db1462f2b46cf27294e91814541616372be76c3ebad465552fb48a57',
    'drawer_solver.py': '413c44eb051c369d23cd5fdd4c577b50b918a983845af560334f3a7e3926a8aa',
}
SEEDS = [11, 23, 47, 83, 131]
# Comparison tolerances are serialization/arithmetic reconciliation only. They do
# not change any task threshold. Source-frame checks use the unchanged thresholds.
METRIC_ATOL = 2e-6
FORCE_ATOL = 2e-5
ANGLE_ATOL_DEG = 1e-5


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def binding(path):
    p = Path(path).resolve(strict=True)
    require(p.is_file(), 'Not a file: ' + str(p))
    return {'path': str(p), 'sha256': sha(p), 'size_bytes': p.stat().st_size}


def same_number(a, b, atol=METRIC_ATOL):
    return (isinstance(a, (int, float)) and not isinstance(a, bool)
            and isinstance(b, (int, float)) and not isinstance(b, bool)
            and math.isfinite(a) and math.isfinite(b) and abs(a-b) <= atol)


def phase_target(t, spec):
    d = spec['phase_durations_s']; settle = d['settle']
    opened = settle + d['open']; held = opened + d['hold_open']
    closed = held + d['close']
    distance = spec['opening_command_m']
    if t < settle:
        return 'settle', 0.0, 0.0
    if t < opened:
        u = (t-settle)/d['open']
        return 'open', distance*(1-math.cos(math.pi*u))/2, distance*math.pi*math.sin(math.pi*u)/(2*d['open'])
    if t < held:
        return 'hold_open', distance, 0.0
    if t < closed:
        u = (t-held)/d['close']
        return 'close', distance*(1+math.cos(math.pi*u))/2, -distance*math.pi*math.sin(math.pi*u)/(2*d['close'])
    return 'hold_closed', 0.0, 0.0


def vector(value, n, name):
    require(isinstance(value, list) and len(value) == n, name + ' shape')
    a = np.asarray(value, dtype=float)
    require(np.isfinite(a).all(), name + ' nonfinite')
    return a


def trace_rows(path, spec):
    rows = []
    with Path(path).open() as f:
        for line in f:
            row = json.loads(line, parse_constant=lambda s: (_ for _ in ()).throw(ValueError('Nonfinite ' + s)))
            require(isinstance(row, dict), 'Trace row is not an object')
            rows.append(row)
    expected = round(sum(spec['phase_durations_s'].values())/spec['dt_s'])
    require(expected == 2460 and len(rows) == expected, 'Expected all 2460 post-step samples')
    for i, row in enumerate(rows):
        require(type(row['step']) is int and row['step'] == i, 'Missing/duplicate/noncontiguous step')
        require(same_number(row['time_s'], (i+1)*spec['dt_s'], 1e-12), 'Post-step timestamp mismatch')
        for name, n in [('drawer_pose', 7), ('payload_pose', 7), ('drawer_velocity', 6),
                        ('payload_velocity', 6), ('applied_force_world_n', 3)]:
            vector(row[name], n, name)
        for name in ['q_m', 'target_q_m']:
            require(isinstance(row[name], (int,float)) and math.isfinite(row[name]), name + ' nonfinite')
        matrix = np.asarray(row['payload_drawer_contact_force_n'], dtype=float)
        require(matrix.shape == (1,1,3) and np.isfinite(matrix).all(), 'Contact impulse matrix shape/nonfinite')
        require(isinstance(row['contacts'], list), 'Contacts must be a list')
        for c in row['contacts']:
            for key in ['p', 'normal', 'impulse']:
                vector(c[key], 3, 'global contact ' + key)
            require(isinstance(c['separation'], (int,float)) and math.isfinite(c['separation']), 'Contact separation nonfinite')
    return rows


def recompute(rows, request, trial):
    """Use independent scalar/vector formulas; frozen code is never imported."""
    spec = request['spec']; dt = spec['dt_s']; limits = spec['limits']; payload = spec['payload']
    durations = spec['phase_durations_s']; settle = durations['settle']
    hold_end = settle + durations['open'] + durations['hold_open']; total = sum(durations.values())
    authored = vector(request['initial_drawer_pose'], 7, 'initial authored pose')
    # Only Z has a redundant native-origin observation. This is a consistency
    # derivation, not an independently recorded initial native state.
    inferred_z = np.asarray([r['drawer_pose'][2]-r['q_m'] for r in rows])
    require(np.ptp(inferred_z) <= 1e-9, 'Inconsistent native Z origin across trace')
    z0 = float(inferred_z[0])
    import_error = trial['import_pose_error_m']
    require(math.isfinite(import_error) and 0 <= import_error <= 1e-3, 'Invalid import translation error')
    require(abs(z0-authored[2]) <= import_error+1e-7, 'Inferred native Z disagrees with import-error bound')
    rng = np.random.default_rng(request['seed']); ctrl = spec['controller']
    native_q=[]; source_q=[]; off=[]; rotations=[]; normalized_rotations=[]
    speeds=[]; spins=[]; drawer_speeds=[]; forces=[]; raw_impulses=[]; penetrations=[]
    retained=True; contact_count=0; initial_drift=0.; open_values=[]; closed_values=[]
    formula_force_error=0.; target_error=0.; quaternion_norm_error=0.; stability=True
    for i,r in enumerate(rows):
        t=i*dt; phase,target,target_v=phase_target(t,spec)
        require(r['phase']==phase, 'Controller phase differs from prescribed pre-step interval')
        target_error=max(target_error,abs(r['target_q_m']-target))
        require(abs(r['target_q_m']-target)<=1e-12,'Controller target differs from prescribed ramp')
        dp=np.asarray(r['drawer_pose'],float); pp=np.asarray(r['payload_pose'],float)
        dv=np.asarray(r['drawer_velocity'],np.float32); pv=np.asarray(r['payload_velocity'],np.float32)
        q=float(dp[2]-z0); native_q.append(q); source_q.append(float(dp[2]-authored[2]))
        require(abs(q-r['q_m'])<=1e-9,'Stored q differs from raw pose/inferred origin')
        off.append(float(np.linalg.norm(dp[:2]-authored[:2])))
        qn=float(np.linalg.norm(dp[3:])); q0n=float(np.linalg.norm(authored[3:]))
        require(qn>0 and q0n>0,'Zero orientation quaternion')
        quaternion_norm_error=max(quaternion_norm_error,abs(qn-1))
        dot=abs(float(np.dot(dp[3:],authored[3:])))
        rotations.append(math.degrees(2*math.acos(min(1.,max(0.,dot)))))
        normalized_rotations.append(math.degrees(2*math.acos(min(1.,dot/(qn*q0n)))))
        speed=max(float(np.linalg.norm(dv[:3])),float(np.linalg.norm(pv[:3])))
        spin=max(float(np.linalg.norm(dv[3:])),float(np.linalg.norm(pv[3:])))
        speeds.append(speed);spins.append(spin);drawer_speeds.append(float(np.linalg.norm(dv[:3])))
        f=np.asarray(r['applied_force_world_n'],np.float32)
        require(abs(float(f[0]))==0 and abs(float(f[1]))==0,'Force is not solely world Z')
        fz=0.
        if t>=settle:
            previous=rows[i-1]
            pre_q=float(previous['drawer_pose'][2]-z0);pre_v=previous['drawer_velocity'][2]
            jitter=rng.uniform(-ctrl['force_jitter_n'],ctrl['force_jitter_n'])
            fz=float(np.clip(ctrl['kp_n_m']*(target-pre_q)+ctrl['kd_ns_m']*(target_v-pre_v)+jitter,-ctrl['force_limit_n'],ctrl['force_limit_n']))
        err=abs(float(f[2])-float(np.float32(fz)));formula_force_error=max(formula_force_error,err)
        require(err<=FORCE_ATOL,'Recorded force differs from seeded bounded PD controller')
        forces.append(abs(fz))
        impulse=float(np.linalg.norm(np.asarray(r['payload_drawer_contact_force_n'],np.float32)))
        raw_impulses.append(impulse)
        penetration=max([0.]+[-c['separation'] for c in r['contacts']]);penetrations.append(penetration)
        if t<settle:initial_drift=max(initial_drift,abs(q))
        if t>=settle:
            relative=pp[:3]-(dp[:3]-authored[:3])
            retained &= bool(np.all(relative>=payload['retained_center_min_relative_to_drawer_translation_m']) and np.all(relative<=payload['retained_center_max_relative_to_drawer_translation_m']))
            contact_count += int(impulse>0.01)
        if hold_end-.5<=t<hold_end:open_values.append(q)
        if total-.5<=t:closed_values.append((abs(q),drawer_speeds[-1]))
        stability &= limits['min_drawer_position_m']<=q<=limits['max_drawer_position_m'] and speed<=limits['max_linear_speed_m_s'] and spin<=limits['max_angular_speed_rad_s']
    metrics={'max_q_m':max(native_q),'max_penetration_m':max(penetrations),
             'max_off_axis_m':max(off),'max_rotation_deg':max(rotations),
             'max_linear_speed_m_s':max(speeds),'max_angular_speed_rad_s':max(spins),
             'payload_contact_samples':contact_count,'payload_retained':retained,
             'max_force_n':max(forces),'initial_drift_m':initial_drift,
             'min_open_hold_q_m':min(open_values),'max_closed_hold_error_m':max(x[0] for x in closed_values),
             'max_closed_hold_speed_m_s':max(x[1] for x in closed_values)}
    checks={'opening':metrics['min_open_hold_q_m']>=spec['opening_required_m'],
            'closing':metrics['max_closed_hold_error_m']<=spec['closure_tolerance_m'],
            'closed_settled':metrics['max_closed_hold_speed_m_s']<=limits['max_final_speed_m_s'],
            'payload_retained':retained,'payload_contact':contact_count>=limits['min_payload_contact_samples'],
            'penetration':metrics['max_penetration_m']<=limits['max_contact_penetration_m'],
            'off_axis':metrics['max_off_axis_m']<=limits['max_drawer_off_axis_m'],
            'rotation':metrics['max_rotation_deg']<=limits['max_drawer_rotation_deg'],
            'initial_drift':initial_drift<=limits['max_initial_drawer_drift_m'],
            'force_bounded':metrics['max_force_n']<=ctrl['force_limit_n']+1e-6,
            'finite_and_stable':bool(stability)}
    comparisons={}
    for name,v in metrics.items():
        observed=trial['metrics'][name]
        if isinstance(v,(bool,int)):
            consistent=type(observed) is type(v) and v==observed
        else:
            tolerance=(ANGLE_ATOL_DEG if name=='max_rotation_deg' else FORCE_ATOL if name=='max_force_n' else METRIC_ATOL)
            # Native XY origin is unrecorded. Its reported translation-error norm
            # provides a triangle-inequality interval, not its direction.
            if name=='max_off_axis_m':tolerance+=import_error
            consistent=same_number(v,observed,tolerance)
        comparisons[name]={'recomputed':v,'reported':observed,'consistent':consistent,
                           'scope':'authored-frame crosscheck; initial native quaternion missing' if name=='max_rotation_deg' else 'authored-frame with import-error bound' if name in ['max_off_axis_m','payload_retained'] else 'raw-trace recomputation'}
    for name,v in checks.items():
        require(type(trial['checks'].get(name)) is bool,'Missing boolean task check '+name)
    # Translational threshold ambiguity is reported separately; no tolerance is
    # added to an acceptance threshold.
    off_interval=[max(0.,max(off)-import_error),max(off)+import_error]
    retention_margin=float('inf')
    for i,r in enumerate(rows):
        if i*dt<settle:continue
        relative=np.asarray(r['payload_pose'][:3])-(np.asarray(r['drawer_pose'][:3])-authored[:3])
        retention_margin=min(retention_margin,float(np.min(relative-np.asarray(payload['retained_center_min_relative_to_drawer_translation_m']))),float(np.min(np.asarray(payload['retained_center_max_relative_to_drawer_translation_m'])-relative)))
    return {'metrics':metrics,'reported_metric_comparison':comparisons,'trace_checks':checks,
            'reported_trace_checks_match':all(trial['checks'][k]==v for k,v in checks.items()),
            'all_reported_metrics_consistent':all(x['consistent'] for x in comparisons.values()),
            'independently_corroborated_failures':[k for k,v in checks.items() if not v and k not in ['off_axis','rotation']],
            'inferred_initial_native_z_m':z0,'reported_import_translation_error_m':import_error,
            'max_source_frame_q_m':max(source_q),'max_normalized_authored_frame_rotation_deg':max(normalized_rotations),
            'max_quaternion_norm_error':quaternion_norm_error,'off_axis_native_initial_interval_m':off_interval,
            'off_axis_gate_unambiguous_under_import_error':not(off_interval[0]<=limits['max_drawer_off_axis_m']<off_interval[1]),
            'payload_retention_min_margin_m':retention_margin,
            'payload_retention_unambiguous_under_import_error':abs(retention_margin)>import_error,
            'max_force_trace_float32_comparison_error_n':formula_force_error,'max_target_error_m':target_error,
            'contact_detector':{'legacy_field':'payload_drawer_contact_force_n','actual_units':'N*s','threshold_raw_impulse_ns':0.01,'threshold_equivalent_force_n':0.01/dt,'max_observed_impulse_ns':max(raw_impulses)},
            'initial_native_xy_and_quaternion_recorded':False}


def encoded(value):
    """Canonical semantic value for read-only USD authored-property comparison."""
    from pxr import Sdf
    if isinstance(value,Sdf.AssetPath):
        return {'asset':value.resolvedPath or value.path}
    if isinstance(value,dict):
        return {str(k):encoded(v) for k,v in sorted(value.items(),key=lambda x:str(x[0]))}
    if value is None or isinstance(value,(str,bool,int,float)):
        return value
    if hasattr(value,'GetReal') and hasattr(value,'GetImaginary'):
        return [float(value.GetReal()),*[float(x) for x in value.GetImaginary()]]
    try:
        return [encoded(x) for x in value]
    except TypeError:
        return str(value)


def source_scene_check(asset, scene_path, request, bindings):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    source=Usd.Stage.Open(str(asset));scene=Usd.Stage.Open(str(scene_path))
    require(source and scene,'USD stage cannot open')
    require(UsdGeom.GetStageMetersPerUnit(source)==1 and UsdGeom.GetStageMetersPerUnit(scene)==1,'Scene units changed')
    require(str(UsdGeom.GetStageUpAxis(source))=='Y' and str(UsdGeom.GetStageUpAxis(scene))=='Y','Scene up axis changed')
    original={str(p.GetPath()):p for p in source.Traverse()};actual={str(p.GetPath()):p for p in scene.Traverse()}
    removed={path for path,p in original.items() if p.IsA(UsdPhysics.Scene)}
    require(set(actual)==(set(original)-removed)|{'/__EvaluatorScene','/__EvaluatorPayload','/__EvaluatorMaterial'},'Unexpected scene prim addition/removal')
    drawer=bindings['drawer_body'];cabinet=bindings['cabinet_body']
    permitted={drawer:{'physxContactReport:threshold','physxRigidBody:solverPositionIterationCount','physxRigidBody:solverVelocityIterationCount','physics:velocity','physics:angularVelocity','physxRigidBody:disableGravity'},cabinet:{'physxContactReport:threshold'}}
    def properties(prim):
        result={}
        for prop in prim.GetAuthoredProperties():
            name=prop.GetName()
            if name.startswith('drive:'):continue
            if isinstance(prop,Usd.Attribute):
                require(prop.GetNumTimeSamples()==0,'Time-sampled scene property '+str(prop.GetPath()))
                result[name]={'type':str(prop.GetTypeName()),'value':encoded(prop.Get()),'connections':[str(x) for x in prop.GetConnections()]}
            else:result[name]={'targets':[str(x) for x in prop.GetTargets()]}
        return result
    compared=0
    for path,p in original.items():
        if path in removed:continue
        q=actual[path];require(p.GetTypeName()==q.GetTypeName() and p.IsActive()==q.IsActive(),'Source prim type/activation changed: '+path)
        expected_apis={a for a in (p.GetMetadata('apiSchemas').GetAppliedItems() if p.GetMetadata('apiSchemas') else []) if not a.startswith('PhysicsDriveAPI:')}
        if path in [drawer,cabinet]:expected_apis.add('PhysxContactReportAPI')
        if path==drawer:expected_apis.add('PhysxRigidBodyAPI')
        require(set(q.GetMetadata('apiSchemas').GetAppliedItems() if q.GetMetadata('apiSchemas') else [])==expected_apis,'Source APIs changed outside declared intervention: '+path)
        left,right=properties(p),properties(q)
        for name in permitted.get(path,set()):left.pop(name,None);right.pop(name,None)
        require(left==right,'Source authored properties changed outside declared intervention: '+path)
        compared+=len(left)
    spec=request['spec'];body=scene.GetPrimAtPath(drawer)
    for name,value in [('physxRigidBody:solverPositionIterationCount',16),('physxRigidBody:solverVelocityIterationCount',8),('physxRigidBody:disableGravity',False)]:
        require(body.GetAttribute(name).Get()==value,'Task body solver setting differs: '+name)
    for name in ['physics:velocity','physics:angularVelocity']:
        require(list(body.GetAttribute(name).Get())==[0,0,0],'Nonzero task initial body velocity')
    for path in [drawer,cabinet,'/__EvaluatorPayload']:
        require(scene.GetPrimAtPath(path).GetAttribute('physxContactReport:threshold').Get()==0,'Contact reporting threshold differs')
    gravity=UsdPhysics.Scene(scene.GetPrimAtPath('/__EvaluatorScene'))
    require(list(gravity.GetGravityDirectionAttr().Get())==[0,-1,0] and same_number(gravity.GetGravityMagnitudeAttr().Get(),9.81,1e-6),'Gravity differs')
    cube=UsdGeom.Cube(scene.GetPrimAtPath('/__EvaluatorPayload'));p=cube.GetPrim();ps=spec['payload']
    require(cube and cube.GetSizeAttr().Get()==1,'Payload is not prescribed unit Cube')
    require(p.HasAPI(UsdPhysics.RigidBodyAPI) and p.HasAPI(UsdPhysics.CollisionAPI),'Payload body/collider missing')
    require(p.GetAttribute('physics:kinematicEnabled').Get() is False,'Payload kinematic')
    require(same_number(p.GetAttribute('physics:mass').Get(),ps['mass_kg'],1e-8),'Payload mass differs')
    rng=np.random.default_rng(request['seed']);position=np.array(ps['initial_center_m'],float)
    position[0]+=rng.uniform(-ps['initial_x_jitter_m'],ps['initial_x_jitter_m']);position[2]+=rng.uniform(-ps['initial_z_jitter_m'],ps['initial_z_jitter_m']);yaw=rng.uniform(-ps['initial_yaw_jitter_deg'],ps['initial_yaw_jitter_deg'])
    require(np.allclose(list(p.GetAttribute('xformOp:translate').Get()),position,atol=1e-12,rtol=0),'Payload seed-dependent position differs')
    require(same_number(p.GetAttribute('xformOp:rotateY').Get(),float(np.float32(yaw)),1e-9),'Payload seed-dependent yaw differs')
    require(np.allclose(list(p.GetAttribute('xformOp:scale').Get()),ps['size_m'],atol=1e-8,rtol=0),'Payload dimensions differ')
    require(list(p.GetAttribute('xformOpOrder').Get())==['xformOp:translate','xformOp:rotateY','xformOp:scale'],'Payload transform stack differs')
    size=np.asarray(ps['size_m']);inertia=ps['mass_kg']/12*(np.dot(size,size)-size*size)
    require(np.allclose(list(p.GetAttribute('physics:diagonalInertia').Get()),inertia,atol=1e-9,rtol=0),'Payload inertia differs')
    require(list(p.GetAttribute('physics:centerOfMass').Get())==[0,0,0],'Payload COM differs')
    require([str(x) for x in p.GetRelationship('material:binding:physics').GetTargets()]==['/__EvaluatorMaterial'],'Payload material binding differs')
    mat=scene.GetPrimAtPath('/__EvaluatorMaterial')
    for name,value in [('physics:staticFriction',ps['surface_friction']),('physics:dynamicFriction',ps['surface_friction']),('physics:restitution',0.)]:
        require(same_number(mat.GetAttribute(name).Get(),value,1e-6),'Payload material differs')
    # The independent fixture must remain free. No scene joint can attach it.
    require(not any('/__EvaluatorPayload' in [str(x) for x in p.GetTargets()] for prim in scene.Traverse() for p in prim.GetRelationships() if p.GetName() in ['physics:body0','physics:body1']),'Payload attached by joint')
    initial=UsdGeom.XformCache().GetLocalToWorldTransform(body);q=initial.ExtractRotationQuat()
    pose=[*initial.ExtractTranslation(),*q.GetImaginary(),q.GetReal()]
    require(np.allclose(pose,request['initial_drawer_pose'],atol=1e-12,rtol=0),'Request authored initial pose differs from scene')
    require(not any(a.startswith('PhysicsDriveAPI:') for prim in scene.Traverse() for a in prim.GetAppliedSchemas()),'Authored drive survived task preparation')
    return {'source_prim_count':len(original),'source_authored_properties_compared':compared,'only_declared_scene_interventions':True,'payload_position_m':position.tolist(),'payload_yaw_deg':float(yaw),'gravity_m_s2':[0,-9.81,0]}


def audit(args):
    root=args.experiment_root.resolve(strict=True);cap=root/'capstone'
    evaluator=args.evaluator.resolve(strict=True);evaluation=args.evaluation.resolve(strict=True)
    bindings_path=args.bindings.resolve(strict=True);asset=args.asset.resolve(strict=True)
    output=args.output.resolve();require(output.is_relative_to(cap),'Audit output must be under capstone')
    require(not output.exists(),'Audit output already exists; create-only')
    output.mkdir(parents=True)
    inputs=[];result={'schema_version':'capstone-independent-task-evidence-audit.v1','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':sha(Path(__file__)),'status':'INCONSISTENT_OR_INCOMPLETE','audit_consistent':False,'task_acceptance_attested':False,'trials':[],'errors':[]}
    def bind(path):
        item=binding(path);inputs.append(item);return item
    try:
        require(sha(asset)==args.expected_asset_sha256,'Unexpected final asset digest')
        for name,digest in FROZEN.items():
            require(sha(evaluator/name)==digest,'Frozen evaluator/spec/solver bytes differ: '+name)
            bind(evaluator/name)
        spec=read(evaluator/'drawer_acceptance.json');require(spec['seeds']==SEEDS,'Prescribed seed list differs')
        b=read(bindings_path);target=Path(b['final_usd']);target=target if target.is_absolute() else bindings_path.parent/target
        require(target.resolve(strict=True)==asset,'Bindings asset path mismatch')
        bind(asset);bind(bindings_path);aggregate_path=evaluation/'report.json';aggregate=read(aggregate_path);bind(aggregate_path)
        expected={'usd_sha256':sha(asset),'bindings_sha256':sha(bindings_path),'protocol_sha256':FROZEN['drawer_acceptance.json'],'evaluator_sha256':FROZEN['drawer_evaluate.py'],'solver_adapter_sha256':FROZEN['drawer_solver.py']}
        require(all(aggregate['inputs'].get(k)==v for k,v in expected.items()),'Aggregate provenance differs from exact frozen inputs')
        require(aggregate['protocol']==spec['protocol_id'],'Aggregate protocol differs')
        require([t.get('seed') for t in aggregate['trials']]==SEEDS,'Expected prescribed five distinct ordered trials')
        structural_path=evaluation/'structural_report.json';structural=read(structural_path);bind(structural_path)
        require(structural['pass'] is True and structural['errors']==[],'Prior structural/source pass is absent')
        # Reuse the already-qualified native dependency identity implementation.
        sys.path.insert(0,str(root/'capstone/repo/agentic/packages/content_agent_workflows'))
        from content_agent_workflows.simready.asset_identity import build_asset_dependency_manifest
        bind(root/'capstone/repo/agentic/packages/content_agent_workflows/content_agent_workflows/simready/asset_identity.py')
        from pxr import Usd
        st=Usd.Stage.Open(str(asset));composed=hashlib.sha256(st.Flatten().ExportToString().encode()).hexdigest()
        require(composed==aggregate['inputs']['composed_stage_sha256'],'Composed source identity differs')
        dep_manifests={'asset':build_asset_dependency_manifest(asset)}
        for d in dep_manifests['asset']['files']:bind(Path(d['path']))
        for seed,reported in zip(SEEDS,aggregate['trials']):
            trial=evaluation/f'seed_{seed}';paths={name:trial/name for name in ['request.json','scene.usda','trial_report.json','trace.jsonl','replay.usda','solver.log']}
            for p in paths.values():bind(p)
            req=read(paths['request.json']);tr=read(paths['trial_report.json'])
            require(tr==reported,'Aggregate/per-seed report differs')
            require(req['seed']==seed and tr['seed']==seed,'Seed mismatch')
            require(req['spec']==spec,'Request controller/payload/threshold spec differs')
            require(Path(req['scene_usd']).resolve(strict=True)==paths['scene.usda'].resolve() and Path(req['output']).resolve()==trial.resolve(),'Request path closure mismatch')
            require(req['scene_sha256']==sha(paths['scene.usda'])==tr['scene_sha256'],'Request/scene/solver-report digest mismatch')
            require(all(req[k]==b[k] for k in ['drawer_body','cabinet_body']) and req['payload_body']=='/__EvaluatorPayload','Request body binding differs')
            require(tr['resolved_contact_sensor_paths']==['/__EvaluatorPayload'] and tr['resolved_contact_filter_paths']==[[b['drawer_body']]],'Recorded contact binding differs')
            require(tr['errors']==[],'Native report has errors')
            scenes=source_scene_check(asset,paths['scene.usda'],req,b)
            dep_manifests[str(seed)]=build_asset_dependency_manifest(paths['replay.usda'])
            for d in dep_manifests[str(seed)]['files']:bind(Path(d['path']))
            rows=trace_rows(paths['trace.jsonl'],spec);computed=recompute(rows,req,tr)
            require(computed['all_reported_metrics_consistent'],'Reported metrics disagree with independent trace formulas')
            require(computed['reported_trace_checks_match'],'Reported trace checks disagree with recomputation')
            require(computed['off_axis_gate_unambiguous_under_import_error'] and computed['payload_retention_unambiguous_under_import_error'],'Import translation uncertainty straddles a check threshold')
            # Interior query records are retained, not replayed or treated as a
            # new initialized query. They were emitted before the first pose read.
            expected_points=[[x,spec['interior_queries']['y_m'],z] for x in spec['interior_queries']['x_m'] for z in spec['interior_queries']['z_m']]
            require([q['position'] for q in tr['interior_queries']]==expected_points,'Interior query point inventory differs')
            interior=not any(q['overlap_count'] for q in tr['interior_queries'])
            require(tr['checks']['interior_clear']==interior,'Interior report contradicts retained query rows')
            require(set(tr['checks'])==set(computed['trace_checks'])|{'interior_clear'},'Unexpected/missing task checks')
            reported_pass=all(tr['checks'].values())
            require(type(tr['pass']) is bool and tr['pass']==reported_pass and tr['status']==('PASS' if reported_pass else 'FAIL'),'Trial verdict contradicts its checks')
            result['trials'].append({'seed':seed,'samples':len(rows),'scene_checks':scenes,'recomputed':computed,'prior_interior_query_result_crosschecked_only':True,'reported_status':tr['status'],'trace_status':'FAIL' if computed['independently_corroborated_failures'] else 'NO_RECOMPUTED_FAILURE'})
        all_pass=all(t['pass'] for t in aggregate['trials']);require(aggregate['pass']==all_pass and aggregate['status']==('PASS' if all_pass else 'FAIL'),'Aggregate task verdict contradicts five reports')
        result.update(status='CONSISTENT_WITH_DECLARED_LIMITS',audit_consistent=True,reported_task_status=aggregate['status'],reported_task_pass=all_pass,trace_failures_corroborated=[{'seed':t['seed'],'checks':t['recomputed']['independently_corroborated_failures']} for t in result['trials']],dependency_manifests=dep_manifests)
    except Exception as e:
        result['errors'].append(type(e).__name__+': '+str(e))
    unique={item['path']:item for item in inputs};result['input_bindings']=list(unique.values())
    result['all_bound_input_bytes_unchanged']=all(binding(Path(x['path']))==x for x in unique.values())
    if not result['all_bound_input_bytes_unchanged']:
        result.update(status='INCONSISTENT_OR_INCOMPLETE',audit_consistent=False);result['errors'].append('Input changed during audit')
    result['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    result['comparison_tolerances']={'metric_serialization_atol':METRIC_ATOL,'force_float32_reconciliation_n':FORCE_ATOL,'source_frame_rotation_comparison_deg':ANGLE_ATOL_DEG,'task_thresholds_modified':False}
    result['formulas']={'physical_sample_time':'(step+1)*dt; controller/window time=step*dt','native_q':'drawer_pose.z - constant inferred from every (pose.z - stored q); inferred origin bounded against authored pose by retained import error','source_off_axis':'norm(drawer_pose.xy - authored_initial.xy); native metric lies within +/- reported import translation error','source_rotation':'degrees(2*acos(clamp(abs(dot(recorded xyzw,authored_initial xyzw)),0,1))); normalized variant also reported, initial native quaternion unrecorded','opening':'minimum q over [hold_open_end-0.5,hold_open_end) >=0.20m','closing':'maximum abs(q) over [total-0.5,total) <=0.02m','settled':'maximum float32 norm(drawer linear velocity) in same closing window <=0.05m/s','payload_retention':'world-axis center box tested after subtracting drawer translation from authored origin, not rotation-aware geometric containment','payload_contact':'count float32 norm(pair impulse matrix)>0.01Ns after settle; force presentation=impulse/dt','penetration':'max(0,-separation) over all scene contact points and all steps','controller':'seeded uniform jitter plus prescribed half-cosine PD target/velocity, clipped to +/-40N; pre-step state from preceding post-step row, float32 trace force compared separately'}
    result['limits']=['Audit consistency is not a new simulation or task/native acceptance. No solver, renderer or model is imported or run.','The initial native XY/quaternion is missing from frozen traces. Native Z is redundantly inferred, source-frame XY is bounded using the retained translation error, and source-frame quaternion crosschecks cannot independently establish the unrecorded native quaternion.','The force-named contact field stores raw impulse Ns. Global contact points omit actor IDs, so penetration cannot be attributed independently to cabinet/drawer/payload pairs. Recorded contact binding paths provide only retained producer provenance.','Interior overlap rows precede first native pose read; this audit does not claim their runtime initialization or recook them. A separately retained initialized positive-controlled preflight is required for clearance claims.','Prior structural/source results are hash-bound and source-to-scene authored properties are compared, but original-source geometric distance checks and real-world material/actuator fidelity are not independently repeated.','Current USD dependency closure and before/after hashes establish retained bytes, not a cryptographic proof that a trusted solver generated the trace. Qualified native execution provenance remains separate.']
    (output/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'audit_consistent':result['audit_consistent'],'status':result['status'],'errors':result['errors'],'audit_sha256':sha(output/'audit.json'),'output':str(output/'audit.json')}))
    return 0 if result['audit_consistent'] else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['experiment-root','evaluation','bindings','asset','evaluator','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-asset-sha256',required=True)
    return audit(p.parse_args())


if __name__=='__main__':
    raise SystemExit(main())
