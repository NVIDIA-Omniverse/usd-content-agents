#!/usr/bin/env python3
"""Run a predeclared matched sequence; no task or evaluator changes here."""
import argparse,datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(os.environ.get('ASTRA_EXPERIMENT_ROOT','/opt/astra-content-value-20260921')).resolve()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def source_check(case):
 data=json.loads((ROOT/'protocol/dataset.json').read_text())
 item=next(a for a in data['assets'] if a['case_id']==case)
 base=ROOT/item['source_root'];bad=[]
 for f in item['original_source_files']:
  p=base/f['path']
  if not p.is_file() or sha(p)!=f['sha256']:bad.append(f['path'])
 if bad:raise RuntimeError('Source changed: '+repr(bad[:10]))
 return {'case':case,'checked_files':len(item['original_source_files']),'source_closure_sha256':item['source_closure_sha256'],'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',required=True);p.add_argument('--run-id',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--cases',nargs='+',required=True);a=p.parse_args()
 queue=ROOT/'runs'/a.run_id/'queues';queue.mkdir(parents=True,exist_ok=True)
 receipt=queue/(a.worker+'.json');result={'worker':a.worker,'gpu':a.gpu,'cases':a.cases,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'running','arms':[]}
 def save():receipt.write_text(json.dumps(result,indent=2)+'\n')
 save()
 try:
  for case in a.cases:
   task=ROOT/'protocol/tasks'/(case+'.json')
   contract=json.loads(task.read_text())
   if contract.get('frozen') is not True:raise RuntimeError('Task not frozen: '+case)
   order=['plain_astra','content_agents'] if int(case[:2])%2 else ['content_agents','plain_astra']
   before=source_check(case)
   for arm in order:
    source_check(case)
    command=[sys.executable,str(ROOT/'scripts/run_arm.py'),'--case',case,'--arm',arm,'--gpu',str(a.gpu),'--budget-seconds','2400','--task-file',str(task),'--run-id',a.run_id]
    step={'case':case,'arm':arm,'task_sha256':sha(task),'started_unix':time.time(),'source_before':before}
    result['current']=step;save()
    cp=subprocess.run(command,cwd=ROOT,check=False)
    step.update(returncode=cp.returncode,ended_unix=time.time(),source_after=source_check(case))
    result['arms'].append(step);save()
    if cp.returncode:raise RuntimeError('Runner infrastructure failure: '+case+'/'+arm)
   result.pop('current',None);save()
  result['status']='complete'
 except Exception as exc:
  result['status']='blocked';result['error']=repr(exc);raise
 finally:
  result['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
if __name__=='__main__':main()
