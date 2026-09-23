"""Wrong edge and reversed endpoint negative controls, no native solver."""
import argparse,copy,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[]
for case in ['08_excavator','10_complex']:
 folder=ROOT/'qualification/local_preparation'/case
 config=json.loads((folder/'measurement/runtime_config.json').read_text());bindings=json.loads((folder/'implementation/selftests'/case/'bindings.json').read_text())
 sys.path.insert(0,str(ROOT/'cases'/case));spec=importlib.util.spec_from_file_location('case_'+case,ROOT/'cases'/case/'case_hooks.py');hooks=importlib.util.module_from_spec(spec);spec.loader.exec_module(hooks)
 checks=hooks.inspect_extra(config,bindings);assert all(c['passed'] for c in checks);rows.append({'case_id':case,'variant':'positive','qualified':True})
 for role in config['contract']['required_joint_roles']:
  for variant in ['wrong_branch','reversed_endpoints']:
   changed=copy.deepcopy(config);j=changed['joints'][changed['joint_roles'][role]]
   if variant=='wrong_branch':j['body0']=j['body1']
   else:j['body0'],j['body1']=j['body1'],j['body0']
   checks=hooks.inspect_extra(changed,bindings);fail=[r['name'] for r in checks if not r['passed']];expected='source_grounded_joint_endpoints:'+role
   assert expected in fail;rows.append({'case_id':case,'variant':variant,'joint_role':role,'qualified':True,'expected_failure':expected,'actual_failures':fail})
(a.output/'summary.json').write_text(json.dumps({'qualified':True,'rows':rows,'model_calls':0,'native_solver_calls':0,'author_output_used':False},indent=2)+'\n');print(len(rows))
