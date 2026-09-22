"""Run native physics positives and independent negative controls remotely."""
import argparse,json,subprocess,sys,copy
from types import SimpleNamespace
import evaluate as evaluator
evaluator.SEEDS=[11] # Synthetic qualification only; frozen benchmark code retains five seeds.
from pathlib import Path
from common import dump
from evaluate import observe
HERE=Path(__file__).resolve().parent

def main():
    subprocess.run([sys.executable,str(HERE/'case_fixtures.py')],check=True)
    reports={}
    for case in ['08_excavator','10_complex']:
        root=HERE/'selftests'/case
        for name in ['positive','kinematic','changed_geometry','wrong_branch','missing_joint']:
            evaluator.evaluate(SimpleNamespace(case=case,usd=root/(name+'.usda'),bindings=root/'bindings.json',inventory=root/'reference/source_inventory.json',output=root/(name+'_result'),source_root=None,solver_python=Path('/opt/astra-content-value-20260921/ovphysx-venv/bin/python'),device='cpu',structural_only=name!='positive'))
            reports[case+':'+name]=json.loads((root/(name+'_result')/'acceptance.json').read_text())
        config=json.loads((root/'positive_result/runtime_config.json').read_text());result=json.loads((root/'positive_result/seed_11.json').read_text())
        if 'trace' in result:
            for mode in ['frozen_joint','broken_closure','missing_load']:
                changed=copy.deepcopy(result)
                if mode=='broken_closure':changed['max_joint_closure_m']=.1
                else:
                    role=config['contract']['required_joint_roles'][-1];p=config['joint_roles'][role]
                    for row in changed['trace']:
                        if mode=='frozen_joint':row['joints'][p]['q_unwrapped']=changed['trace'][0]['joints'][p]['q_unwrapped']
                        else:row['joints'][p]['external_effort']=0
                checks=observe(changed,config);failed=[x['name'] for x in checks if not x['passed'] and not x.get('insufficient',False)]
                reports[case+':observer_'+mode]={'accepted':not bool(failed),'concrete_failures':failed}
        dump(HERE/'selftest_report.json',reports)
    summary={k:{'accepted':v['accepted'],'concrete_failures':v['concrete_failures']} for k,v in reports.items()}
    dump(HERE/'selftest_summary.json',summary)
    assert all(v['accepted'] if k.endswith(':positive') else not v['accepted'] and bool(v['concrete_failures']) for k,v in reports.items()),summary
    print('ALL SYNTHETIC POSITIVE AND NEGATIVE CHECKS PASSED',flush=True)

if __name__=='__main__':main()
