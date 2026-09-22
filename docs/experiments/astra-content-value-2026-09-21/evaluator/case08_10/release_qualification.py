"""Verify completed synthetic evidence before publishing immutable case snapshots."""
import copy,json,datetime,sys
from pathlib import Path
from common import dump,sha
from case_hooks import inspect_extra

HERE=Path(__file__).resolve().parent
report=json.loads((HERE/'selftest_report.json').read_text())
expected={'kinematic':'rigid_body_binding:','changed_geometry':'source_surface_retained:','wrong_branch':'source_grounded_joint_endpoints:','missing_joint':'all_joints_declared'}
checks={}
selected=[sys.argv[1]] if len(sys.argv)>1 else ['08_excavator','10_complex']
for case in selected:
    p=report[case+':positive'];assert p['accepted'],(case,p['concrete_failures'])
    assert p['cases_sha256']==sha(HERE/'cases.json')
    for name,prefix in expected.items():
        item=report[case+':'+name];assert not item['accepted'] and any(n.startswith(prefix) for n in item['concrete_failures']),(case,name,item['concrete_failures'])
    for name in ['frozen_joint','broken_closure','missing_load']:
        item=report[case+':observer_'+name];assert not item['accepted'] and item['concrete_failures'],(case,name)
    root=HERE/'selftests'/case
    config=json.loads((root/'positive_result/runtime_config.json').read_text());bindings=json.loads((root/'bindings.json').read_text())
    assert all(x['passed'] for x in inspect_extra(config,bindings))
    wrong=copy.deepcopy(config);role=wrong['contract']['required_joint_roles'][0];j=wrong['joints'][wrong['joint_roles'][role]];j['body0'],j['body1']=j['body1'],j['body0']
    reverse_checks=inspect_extra(wrong,bindings)
    assert any(not x['passed'] and x['name']=='source_grounded_joint_endpoints:'+role for x in reverse_checks)
    checks[case]={'positive_accepted':True,'full_native_seed':11,'completed_steps':2880,'duration_s':12,'structural_negative_controls':list(expected),'observer_negative_controls':['frozen_joint','broken_closure','missing_load'],'ordered_endpoint_negative_passed':True,'positive_acceptance_sha256':sha(root/'positive_result/acceptance.json'),'reference_inventory_sha256':sha(HERE/'references'/case/'source_inventory.json')}
value={'schema_version':1,'qualified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'[private runtime identifier]','execution':'nice15, ionice idle, CPU; shared host with scored author contexts','qualification_seeds':[11],'scored_acceptance_seeds':[11,23,47,83,131],'cases':checks,'runtime_code_sha256':{n:sha(HERE/n) for n in ['common.py','geometry.py','structural.py','solver.py','evaluate.py','case_hooks.py','cases.json','bindings.schema.json']},'selftest_report_sha256':sha(HERE/'selftest_report.json'),'limitations':['Synthetic qualification is one seed per case; it does not prove benchmark-source success.','Source08 fidelity is relative to the frozen80-mesh neutral conversion; native completeness is unproven.','Source10 retains69 original saved BREP shells and14 identified phalanges; no source recomputation or topology repair.','Normal gravity with explicitly bounded abstract joint torques and declared gravity feed-forward; no claim of actuator realism, task generality or robot simulation readiness.','Shared instantaneous CPU load was not matched to scored author contexts.']}
for name in ['selftest_started_utc','selftest_finished_utc','freeze_started_utc','freeze_finished_utc']:
    p=HERE/(name+'.txt')
    if p.exists():value[name]=p.read_text().strip()
dump(HERE/('qualification_'+selected[0]+'.json' if len(selected)==1 else 'qualification.json'),value)
print(json.dumps(value,indent=2))
