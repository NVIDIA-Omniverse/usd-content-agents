"""Synthetic orchestration regression; never a scored source case."""
import hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
root=HERE/'tests/non_submission_fixture'
assert not root.exists(),'Preserve previous regression evidence'
for arm in ['plain_astra','content_agents']:
    run=root/'runs/synthetic-qualification/03_hinge'/arm
    result=root/'evaluations/synthetic-qualification/03_hinge'/arm
    run.mkdir(parents=True);result.mkdir(parents=True)
    (run/'execution.json').write_text('{}\n')
    (run/'output_manifest.json').write_text('{"files":[]}\n')
    (run/'submission.json').write_text('{"claimed_accepted":false,"final_scene":null,"bindings":null}\n')
    (result/'acceptance.json').write_text(json.dumps({'accepted':False,'status':'not_accepted','concrete_failures':['submission_contract'],'inconclusive_checks':[],'scope':'Synthetic contractual non-submission fixture; no real source'})+'\n')
command=[sys.executable,str(HERE/'worker.py'),'--root',str(root),'--run-id','synthetic-qualification','--case','03_hinge','--arm','both','--qualification',str(HERE/'qualification_r2.json')]
subprocess.run(command,check=True)
records=[]
for arm in ['plain_astra','content_agents']:
    v1=root/'evaluations/synthetic-qualification/03_hinge'/arm/'acceptance.json'
    out=root/'evaluation_adjudications/synthetic-qualification/03_hinge'/arm/'source_distance_v1_1/acceptance.json'
    d=json.loads(out.read_text())
    good=d['adjudication_action']=='skipped_non_submission' and d['status']=='not_accepted' and d['concrete_failures']==['submission_contract'] and not d['accepted'] and d['source_surface_checks_replaced']==0 and d['v1_acceptance_sha256']==hashlib.sha256(v1.read_bytes()).hexdigest()
    records.append({'arm':arm,'passed':good,'receipt':str(out)})
report={'scope':'Unscored null-submission batch orchestration regression; no source asset or physics','all_passed':all(x['passed'] for x in records),'tests':records,'worker_sha256':hashlib.sha256((HERE/'worker.py').read_bytes()).hexdigest()}
(HERE/'non_submission_qualification.json').write_text(json.dumps(report,indent=2)+'\n')
assert report['all_passed'];print(json.dumps(report))
