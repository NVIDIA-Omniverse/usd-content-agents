"""Sequential host scheduling only; numerical logic stays in qualified worker."""
import argparse,datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import worker
HERE=Path(__file__).resolve().parent
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main(a):
    if os.getpriority(os.PRIO_PROCESS,0)<15:os.nice(15-os.getpriority(os.PRIO_PROCESS,0))
    out=a.root/'evaluation_adjudications'/a.run_id/'_dispatch';out.mkdir(parents=True,exist_ok=True)
    receipt=out/(a.label+'.json');logpath=out/(a.label+'.log')
    assert not receipt.exists() and not logpath.exists(),'Refusing to overwrite dispatcher evidence'
    pending={(c,arm) for c in a.cases for arm in ['content_agents','plain_astra']}
    record={'scope':'Sequential host scheduling for versioned post-freeze adjudications','label':a.label,'cases':a.cases,'started_utc':utc(),'worker_sha256':worker.sha(HERE/'worker.py'),'qualification_sha256':worker.sha(HERE/'qualification_r2.json'),'dispatcher_sha256':worker.sha(__file__),'jobs':[],'status':'running','pid':os.getpid()}
    def save():
        record['pending']=sorted('/'.join(x) for x in pending);record['updated_utc']=utc();receipt.write_text(json.dumps(record,indent=2)+'\n')
    save();deadline=time.monotonic()+a.wait_seconds
    with logpath.open('w') as log:
        while pending:
            progress=False
            for case,arm in sorted(pending):
                dest=a.root/'evaluation_adjudications'/a.run_id/case/arm/worker.VERSION
                if (dest/'acceptance.json').is_file():
                    record['jobs'].append({'case':case,'arm':arm,'action':'already_completed','receipt_sha256':worker.sha(dest/'acceptance.json')});pending.remove((case,arm));progress=True;save();continue
                if dest.exists():continue  # Another owner or preserved incomplete attempt needs attention.
                run=a.root/'runs'/a.run_id/case/arm;v1=a.root/'evaluations'/a.run_id/case/arm/'acceptance.json'
                if not all(p.is_file() for p in [run/'execution.json',run/'output_manifest.json',v1]):continue
                submission=worker.load(run/'submission.json') if (run/'submission.json').is_file() else {}
                null=any(not isinstance(submission.get(k),str) or not submission[k].strip() for k in ['final_scene','bindings'])
                snapshot,inventory=worker.discover(a.root,case,a.run_id)
                if not null and (not snapshot.is_dir() or not inventory.is_file()):continue
                command=[sys.executable,str(HERE/'worker.py'),'--root',str(a.root),'--run-id',a.run_id,'--case',case,'--arm',arm,'--qualification',str(HERE/'qualification_r2.json'),'--fresh-conveyor-if-needed']
                job={'case':case,'arm':arm,'started_utc':utc(),'command':command};record['jobs'].append(job);save()
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT);log.flush()
                job.update(returncode=result.returncode,finished_utc=utc())
                if result.returncode:
                    job['status']='failed_or_requires_review';pending.remove((case,arm))
                else:
                    job['status']='completed';job['receipt_sha256']=worker.sha(dest/'acceptance.json');pending.remove((case,arm))
                progress=True;save()
            if not pending:break
            if time.monotonic()>=deadline:break
            if not progress:time.sleep(min(10,max(.1,deadline-time.monotonic())))
    record.update(status='complete' if not pending and all(x.get('returncode',0)==0 for x in record['jobs']) else 'incomplete_or_requires_review',finished_utc=utc());save()
    print(json.dumps({'receipt':str(receipt),'status':record['status'],'pending':record['pending']}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/opt/astra-content-value-20260921'));p.add_argument('--run-id',default='pilot-v1');p.add_argument('--cases',nargs='+',choices=worker.CASES,required=True);p.add_argument('--label',required=True);p.add_argument('--wait-seconds',type=float,default=14400);main(p.parse_args())
