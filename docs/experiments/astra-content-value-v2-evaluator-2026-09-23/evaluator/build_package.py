"""Build v2 only from retained evaluator code and independent source references.

No author asset, author script, author trace or capstone asset is an input.
This is a development build, never an automatic freeze.
"""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
V1 = HERE.parents[1] / 'astra-content-value-20260921'
IDS = ['02_conveyor', '03_hinge', '04_gripper', '05_vise', '06_engine',
       '07_robot_arm', '08_excavator', '09_printer', '10_complex']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2) + '\n')


def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise ValueError('Patch anchor changed: ' + old[:90] + ' count=' + str(text.count(old)))
    return text.replace(old, new)


def build():
    provenance = []
    for ident in IDS:
        if ident == '02_conveyor':
            origin = V1 / 'evaluator/conveyor'
            reference = origin / 'reference'
            inventory = origin / 'source_inventory.json'
        elif ident in ('08_excavator', '10_complex'):
            origin = V1 / 'evaluator/case08_10/frozen' / ident
            reference = V1 / 'evaluator/case08_10/references' / ident
            inventory = reference / 'source_inventory.json'
        else:
            origin = V1 / 'evaluator/general/frozen' / ident
            reference = V1 / 'evaluator/general/references' / ident
            if ident == '04_gripper':
                reference = V1 / 'evaluations/pilot-v1/04_gripper/_runtime/references/04_gripper'
            inventory = reference / 'source_inventory.json'
        target = HERE / 'cases' / ident
        if (target / 'v2_freeze.json').exists():
            raise RuntimeError('Refusing to overwrite frozen v2 case: ' + ident)
        target.mkdir(parents=True, exist_ok=True)
        for p in origin.iterdir():
            if p.is_file() and p.suffix in ('.py', '.json'):
                # Original freeze records belong to v1 and must not purport to
                # freeze modified code. Their hashes stay in provenance only.
                if p.name in ('frozen_manifest.json', 'freeze.json', 'source_inventory.json'):
                    continue
                shutil.copy2(p, target / p.name)
                provenance.append({'case_id': ident, 'kind': 'v1_evaluator',
                                   'path': str(p.relative_to(V1)), 'sha256': sha(p)})
        refout = target / 'reference'
        refout.mkdir(exist_ok=True)
        inv = json.loads(inventory.read_text())
        expected = None
        manifest = origin / 'frozen_manifest.json'
        if manifest.exists():
            expected = json.loads(manifest.read_text())['reference_inventory_sha256']
        elif (origin / 'freeze.json').exists():
            expected = json.loads((origin / 'freeze.json').read_text())['source_inventory_sha256']
        assert not expected or sha(inventory) == expected, ident + ' inventory hash differs from v1 freeze'
        shutil.copy2(inventory, refout / 'source_inventory.json')
        for part in inv['parts']:
            p = reference / part['geometry_file']
            assert sha(p) == part['geometry_sha256'], str(p)
            dest = refout / part['geometry_file']; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        provenance.append({'case_id': ident, 'kind': 'independent_source_reference',
                           'path': str(inventory.relative_to(V1)), 'sha256': sha(inventory),
                           'part_count': len(inv['parts']), 'all_geometry_hashes_verified': True})
        for p in (HERE / 'common').glob('v2_*.py'):
            shutil.copy2(p, target / p.name)
        fixtures = {'02_conveyor': V1/'evaluator/conveyor/fixtures.py',
                    '03_hinge': V1/'evaluator/general/fixtures.py',
                    '04_gripper': V1/'evaluator/general/case04_fixtures.py',
                    '05_vise': V1/'evaluator/general/fixtures.py',
                    '06_engine': V1/'evaluator/general/fixture_engine.py',
                    '07_robot_arm': V1/'evaluator/general/fixture_arm.py',
                    '08_excavator': V1/'evaluator/case08_10/case_fixtures.py',
                    '09_printer': V1/'evaluator/general/fixture_axes.py',
                    '10_complex': V1/'evaluator/case08_10/case_fixtures.py'}
        fixture_path = HERE/'qualification'/'fixture_builders'/(ident+'.py')
        fixture_path.parent.mkdir(exist_ok=True)
        shutil.copy2(fixtures[ident],fixture_path)
        provenance.append({'case_id':ident,'kind':'synthetic_fixture_builder','path':str(fixtures[ident].relative_to(V1)),'sha256':sha(fixtures[ident])})
        evaluator = target / 'evaluate.py'
        evaluator.write_text(evaluator.read_text().replace("check(checks,'reference_inventory_frozen',sha(args.inventory)==frozen['reference_inventory_sha256'])", "check(checks,'reference_inventory_frozen',sha(args.inventory)==frozen['reference_inventory_sha256'],insufficient=True)"))
        evaluator_text = evaluator.read_text()
        evaluator_text = 'from v2_verdict import classify_uncalibrated_runtime, witness_check\n' + evaluator_text
        evaluator_text = evaluator_text.replace('report=verdict(checks)', 'checks=classify_uncalibrated_runtime(checks);report=verdict(checks)')
        evaluator_text = evaluator_text.replace("            if 'trace' in unloaded and 'trace' in loaded:\n", "            checks.append(witness_check(unloaded,'paired_seed11'))\n            if 'trace' in unloaded and 'trace' in loaded:\n")
        evaluator.write_text(evaluator_text)
        p = target / 'geometry.py'
        text = p.read_text()
        # Preserve independent importer code verbatim. Only the comparison
        # implementation is replaced, with the original retained in provenance.
        start = text.index('def surface_compare(')
        end = text.find("\nif __name__", start)
        if end < 0:
            end = len(text)
        text = text[:start] + 'from v2_geometry import surface_compare\n' + text[end:]
        p.write_text(text)
        p = target / 'structural.py'; text = p.read_text()
        text = text.replace('import numpy as np', 'import numpy as np\nimport json, time\nfrom v2_policy import auxiliary_checks\nfrom v2_process import InspectionDeadline')
        text = replace(text, "    body_roles = {b['role']:b['path'] for b in body_entries}",
                       "    body_roles = {b['role']:b['path'] for b in body_entries}\n    check(checks,'required_body_roles_unique',all(sum(e['role']==r for e in body_entries)==1 for r in contract['required_body_roles']))")
        if 'moving_body_has_source_geometry:' in text:
            text = replace(text, "    for path,body in bodies.items():\n        check(checks,'moving_body_has_source_geometry:'+path,not body['moving'] or any(m['body_path']==path for m in source_map))",
                           "    for name,passed,evidence in auxiliary_checks(bindings,bodies,joints,source_map,contract):\n        check(checks,name,passed,evidence)")
        text = replace(text, "        okay,detail=surface_compare(expected,source['faces'],v,f,tol)\n        check(checks,'source_surface_retained:'+ident,okay,detail)",
                       "        started=time.monotonic()\n        with (output/'source_progress.jsonl').open('a') as progress:progress.write(json.dumps({'source_id':ident,'state':'started'})+'\\n')\n        try:\n            okay,detail=surface_compare(expected,source['faces'],v,f,tol)\n            check(checks,'source_surface_retained:'+ident,okay,detail)\n        except InspectionDeadline:raise\n        except Exception as exc:\n            okay=False;detail={'error':type(exc).__name__+': '+str(exc)}\n            check(checks,'source_measurement_completed:'+ident,False,detail,insufficient=True)\n        with (output/'source_progress.jsonl').open('a') as progress:progress.write(json.dumps({'source_id':ident,'state':'completed','passed':okay,'elapsed_s':time.monotonic()-started,'measurement':detail})+'\\n')")
        text = replace(text, 'cube.AddTranslateOp().Set(Gf.Vec3d(1000,1000,1000))', 'cube.AddTranslateOp().Set(Gf.Vec3d(1,1,1))')
        text = replace(text, '    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())',
                       "    # Collision-free evaluator witness avoids source interactions at any pose.\n    cube.CreateVisibilityAttr('invisible')")
        text = replace(text, "    witness='/__EvaluatorGravityWitness'", "    witness='/__EvaluatorGravityWitness'\n    if stage.GetPrimAtPath(witness):raise ValueError('Reserved evaluator witness path already exists')")
        text = text.replace('# Evaluator-owned free-fall witness, remote from asset; never writes original.', '# Evaluator-owned collision-free gravity witness near origin; never writes original.')
        text = replace(text,"check(checks,'source_bytes:'+f['path'],p.is_file() and sha(p)==f['sha256'])","check(checks,'source_bytes:'+f['path'],p.is_file() and sha(p)==f['sha256'],insufficient=True)")
        text = replace(text,"check(checks,'frozen_geometry_hash:'+ident,sha(pfile)==part['geometry_sha256'])","check(checks,'frozen_geometry_hash:'+ident,sha(pfile)==part['geometry_sha256'],insufficient=True)")
        p.write_text(text)
        p = target / 'evaluate.py'; text = p.read_text()
        text = 'from v2_process import run as run_isolated, inspect_with_deadline\nfrom v2_inputs import author_input_preflight\n' + text
        signature='def evaluate(a):' if ident=='02_conveyor' else 'def evaluate(args):'
        argument='a' if ident=='02_conveyor' else 'args'
        text=replace(text,signature,signature+'\n '+('' if ident=='02_conveyor' else '   ')+'if not author_input_preflight('+argument+',HERE):return')
        text = text.replace('subprocess.run(', 'run_isolated(')
        text = text.replace('=inspect(', '=inspect_with_deadline(inspect,')
        text = text.replace("'expected_vz':expected})", "'expected_vz':expected},True)")
        if ident == '02_conveyor':
            text = replace(text, "check(checks,f'seed{seed}:'+name,passed)", "check(checks,f'seed{seed}:'+name,passed,insufficient=(name=='independent_gravity'))")
        p.write_text(text)
        if ident == '04_gripper':
            p = target / 'case04_evaluate.py'; text = p.read_text()
            text = replace(text, "chk('gripper_independent_gravity_witness',good,w)", "chk('gripper_independent_gravity_witness',good,w,True)")
            p.write_text(text)
        else:
            p = target / 'solver.py'; text = p.read_text()
            if '*effort_jitter' in text:
                text = 'from v2_control import clamp_effort\n' + text
                text = replace(text, '*effort_jitter\n', "*effort_jitter\n                effort=clamp_effort(effort,setting['max_effort'])\n")
            if "trace.append({'t':t,'poses':poses" in text:
                if 'initial_poses=' not in text:
                    text = replace(text, '        poses=poses_now()\n', '        poses=poses_now()\n        initial_poses={p:list(q) for p,q in poses.items()}\n')
                text = text.replace("trace.append({'t':t,'poses':poses", "trace.append({'t':t,'pose_time_s':t,'velocity_time_s':(step+1)*dt,'control_time_s':t,'poses':poses")
                text = replace(text, "'gravity_witness':{'initial_pose':witness_initial,'at_point_one_s':witness_at_point_one},",
                               "'native_initial_poses':initial_poses,'trace_timing':'poses/joints/control at t; velocities/contacts after step at t+dt',\n                'gravity_witness':{'initial_pose':witness_initial,'at_point_one_s':witness_at_point_one},")
            p.write_text(text)
        p = target / 'bindings.schema.json'; schema = json.loads(p.read_text())
        if ident == '06_engine':
            schema['properties']['bodies']['items']['properties']['auxiliary'] = {'const': 'passive_constraint'}
        dump(p, schema)
        if ident == '06_engine':
            p = target / 'cases.json'; data = json.loads(p.read_text())
            data['cases'][ident]['allow_passive_auxiliary_bodies'] = True
            dump(p, data)
        task = json.loads((V1 / 'protocol/tasks' / (ident + '.json')).read_text())
        task = json.loads(json.dumps(task).replace('/opt/astra-content-value-20260921','/opt/astra-content-value-20260922-rerun'))
        task = json.loads(json.dumps(task).replace('/opt/astra-content-value-20260922-rerun/protocol/tasks/'+ident+'_source_inventory.json','/input/'+ident+'_source_inventory.json'))
        task.pop('evaluator_freeze_sha256',None)
        task['source_inventory_file']='/input/'+ident+'_source_inventory.json'
        task['source_inventory_sha256']=sha(V1/'protocol/tasks'/(ident+'_source_inventory.json'))
        if ident=='02_conveyor':task['private_reference_inventory_sha256']=task.pop('reference_inventory_sha256')
        task['frozen'] = False
        task['protocol_id'] = 'astra-content-value-rerun-v2'
        task['v2_measurement_changes'] = [
            'All original source identities, grounded roles, force/torque controllers, task thresholds and five seeds retained.',
            'Recentered millimetre nearest-triangle queries; conservative all-surface triangle-correspondence fast path; unchanged area/bounds tolerances.',
            'Exact correspondence also retains every loose tessellation point; an unchanged original off-surface point is no longer incorrectly required to lie on a triangle.',
            'Source progress retained; measurement/worker timeout is INCONCLUSIVE, never an author failure or a pass.',
            'Collision-free evaluator gravity witness moved to (1,1,1) metres; witness failure is evaluator insufficiency; actual authored Earth gravity remains a concrete criterion.'
            ,'Controller effort is clamped after the same seeded0.98–1.02 jitter, making the published actuator cap a hard limit. PD gains, distribution, targets, separate external test load and all acceptance thresholds remain unchanged.'
        ]
        if ident == '06_engine':
            task['v2_measurement_changes'].append('Explicitly declared passive source-free constraint bodies permitted only for case06: at most eight, each joined to two distinct neighbors, unactuated, never a required source role. All are observed and need finite positive native dynamics and colliders. All original parts remain mapped exactly once.')
            task['submission_contract'] = task['submission_contract'].split('Bindings JSON schema:')[0] + 'Source-free passive constraint bodies may be declared with auxiliary="passive_constraint" under the exact v2 policy below. They never replace required source geometry or receive a controlled joint effort. Bindings JSON schema:\n' + json.dumps(schema,separators=(',',':'))
            task['public_acceptance']['task_parameters']['allow_passive_auxiliary_bodies'] = True
            task['public_acceptance']['task_parameters']['passive_auxiliary_policy'] = {'max_count':8,'declaration':'auxiliary: passive_constraint','minimum_distinct_joint_neighbors':2,'controlled_incident_joint_permitted':False,'required_original_body_role_permitted':False,'all_native_states_observed':True}
        if ident in ('02_conveyor','04_gripper'):
            task['v2_measurement_changes']=[entry for entry in task['v2_measurement_changes'] if not entry.startswith('Controller effort is clamped')]
        if ident=='02_conveyor':
            task['public_acceptance']['normalization']=task['public_acceptance']['normalization'].replace('a distant free-fall witness','a collision-free free-fall witness at (1,1,1)m')
        task['public_acceptance']['measurement_version'] = 2
        task['public_acceptance']['input_failure_policy'] = 'Missing, empty, malformed or schema-invalid author artifacts are concrete task rejection (FAIL). Evaluator infrastructure, reference availability, native runtime failure or measurement timeout is INCONCLUSIVE unless an independent concrete criterion failed.'
        task['public_acceptance']['runtime_calibration_policy'] = 'A failed independent gravity witness makes failed physics criteria from that same seed INCONCLUSIVE, including paired engine response when either paired run is uncalibrated. Raw observations and failed predicates are retained. Independent static/source/schema failures remain concrete FAIL.'
        task['public_acceptance']['source_metric'] = 'All original instances, unchanged bounds and 3% area thresholds. Conservative triangle correspondence proves every surface point within tolerance and preserves all loose points; otherwise symmetric all-vertex plus deterministic surface-sample nearest-triangle distances in recentered millimetres. This remains geometry preservation, not exact topology or materials equivalence.'
        task['public_acceptance']['runtime_budgets_s'] = {'input_stage_load':900,'complete_source_stage_inspection':1800,'each_native_trial':300 if ident=='02_conveyor' else 900,'engine_paired_unloaded_trial':900}
        task['public_acceptance']['witness_policy'] = 'A collision-free independently instrumented body at (1,1,1)m must show expected freefall within4mm displacement and0.03m/s velocity at0.1s; witness failure means evaluator insufficiency. Authored gravity must independently be Earth gravity. No witness writes to the submitted asset.'
        task['private_evaluator_snapshot'] = 'evaluator/cases/'+ident
        task['private_evaluator_snapshot_sha256'] = None
        task['private_evaluator_cases_sha256'] = sha(target/('contract.json' if ident=='02_conveyor' else 'cases.json'))
        task['freeze_basis'] = 'UNFROZEN v2 candidate. Original independent source hashes retained; v2 local controls and fresh remote native controls must qualify before either scored arm.'
        task['evaluator_qualification'] = {'frozen': False, 'native_requalification': 'required_before_scored_runs'}
        dump(HERE / 'tasks' / (ident + '.json'), task)
        shutil.copy2(V1 / 'protocol/tasks' / (ident + '_source_inventory.json'), HERE / 'tasks' / (ident + '_source_inventory.json'))
    dump(HERE / 'provenance.json', {'schema_version': 1, 'status': 'development_unfrozen', 'inputs': provenance,
                                  'author_outputs_used_as_inputs': False})


if __name__ == '__main__':
    build()
