#!/usr/bin/env python3
"""Run frozen independent evaluators after author outputs finish; never repair them."""
import argparse, hashlib, json, os, subprocess, time, traceback
from pathlib import Path

ROOT=Path(os.environ.get('ASTRA_EXPERIMENT_ROOT','/opt/astra-content-value-20260921'))
def read(p): return json.loads(p.read_text())
def write(p,d): p.write_text(json.dumps(d,indent=2)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def inside(run,value):
    if not isinstance(value,str) or not value:return None
    p=(run/value).resolve()
    if not p.is_relative_to(run.resolve()) or not p.is_file():return None
    return p
def reject(out,reason):
    out.mkdir(parents=True,exist_ok=False)
    write(out/'acceptance.json',{'status':'not_accepted','accepted':False,'concrete_failures':['submission_contract'],'inconclusive_checks':[],'reason':reason,'scope':'Contractual non-delivery; no physical acceptance inferred and no author output changed.'})
def evaluate(case,arm,run_id):
    run=ROOT/'runs'/run_id/case/arm;out=ROOT/'evaluations'/run_id/case/arm
    if out.exists():return
    if not (run/'output_manifest.json').exists():return
    start=time.time();submission=read(run/'submission.json') if (run/'submission.json').exists() else {}
    scene=inside(run,submission.get('final_scene'));bindings=inside(run,submission.get('bindings'))
    if not scene or not bindings:
        reject(out,'Missing or null submitted final_scene/bindings within the completed run directory.');return
    before={'scene_sha256':sha(scene),'bindings_sha256':sha(bindings),'output_manifest_sha256':sha(run/'output_manifest.json')}
    py=ROOT/'repo/.venv/bin/python';vendor=ROOT/'evaluator/general/vendor'
    if case=='02_conveyor':
        base=ROOT/'evaluator/conveyor'
        command=[str(py),str(base/'evaluate.py'),'--usd',str(scene),'--bindings',str(bindings),'--inventory',str(base/'reference/source_inventory.json'),'--source-root',str(ROOT/'assets'/case),'--output',str(out)]
    elif case in ('08_excavator','10_complex'):
        base=ROOT/'evaluator/case08_10';frozen=base/'frozen'/case
        command=[str(py),str(frozen/'evaluate.py'),'--usd',str(scene),'--bindings',str(bindings),'--inventory',str(base/'references'/case/'source_inventory.json'),'--case',case,'--output',str(out)]
    else:
        base=ROOT/'evaluator/general';frozen=base/'frozen'/case
        command=[str(py),str(frozen/'evaluate.py'),'--usd',str(scene),'--bindings',str(bindings),'--inventory',str(base/'references'/case/'source_inventory.json'),'--case',case,'--output',str(out)]
    logdir=out.parent;logdir.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(vendor),CUDA_VISIBLE_DEVICES='')
    with (logdir/(arm+'-independent.log')).open('w') as log:
        proc=subprocess.run(['nice','-n','15','ionice','-c','3',*command],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=5400)
    out.mkdir(parents=True,exist_ok=True)
    after={'scene_sha256':sha(scene),'bindings_sha256':sha(bindings),'output_manifest_sha256':sha(run/'output_manifest.json')}
    write(out/'evaluation_launch.json',{'case_id':case,'arm':arm,'argv':command,'started_unix':start,'ended_unix':time.time(),'returncode':proc.returncode,'before':before,'after':after,'author_artifacts_unchanged':before==after,'device':'cpu','compute_environment':'same Horde node; independent native PhysX evaluator; nice15 idle I/O'})
    if not (out/'acceptance.json').exists() and not (out/'report.json').exists():
        write(out/'acceptance.json',{'status':'inconclusive','accepted':False,'concrete_failures':[],'inconclusive_checks':['evaluator_did_not_produce_verdict'],'returncode':proc.returncode})
def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',required=True);p.add_argument('--run-id',default='pilot-v1');p.add_argument('--worker',required=True);a=p.parse_args()
    status=ROOT/'evidence'/(a.worker+'-evaluation-worker.json');completed=[]
    while True:
        for case in a.cases:
            for arm in ('plain_astra','content_agents'):
                key=case+'/'+arm
                if key in completed:continue
                out=ROOT/'evaluations'/a.run_id/case/arm
                if out.exists() and any((out/x).exists() for x in ('acceptance.json','report.json')):completed.append(key);continue
                run=ROOT/'runs'/a.run_id/case/arm
                if not (run/'output_manifest.json').exists():continue
                family=ROOT/'evaluator'/('case08_10' if case in ('08_excavator','10_complex') else 'general')
                if case!='02_conveyor' and (not (family/'frozen'/case/'evaluate.py').is_file() or not (family/'references'/case/'source_inventory.json').is_file()):continue
                write(status,{'status':'evaluating','current':key,'completed':completed,'updated_unix':time.time()})
                try:evaluate(case,arm,a.run_id)
                except Exception:
                    err=ROOT/'evidence'/(a.worker+'-'+case+'-'+arm+'-evaluation-error.json')
                    write(err,{'error':traceback.format_exc(),'at_unix':time.time()})
                    # Do not silently retry a partially written evaluator directory.
                    write(status,{'status':'blocked','current':key,'completed':completed,'error_file':str(err)});raise
                completed.append(key)
        done=len(completed)==2*len(a.cases)
        write(status,{'status':'complete' if done else 'waiting_completed_authoring','completed':completed,'updated_unix':time.time()})
        if done:return
        time.sleep(10)
if __name__=='__main__':main()
