"""Create fresh private dispatch configuration from a complete protocol freeze."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'harness'))
from common import read, require, sha256, validate_protocol, write_new
from run_queue import validate_jobs


def build(destination, templates, hosts):
    destination=Path(destination).resolve()
    require(not destination.exists(),'Fresh controller directory required')
    freeze=read(ROOT/'protocol/freeze.json')
    require(freeze.get('frozen') is True and freeze.get('status')=='PASS','Final freeze required')
    for row in freeze['files']:
        require(sha256(ROOT/row['path'])==row['sha256'],'Frozen file changed: '+row['path'])
    frozen={row['path']:row['sha256']for row in freeze['files']}
    templates=Path(templates).resolve()
    for name in ('manifest.json','node0.json','node1.json'):
        relative=str((templates/name).relative_to(ROOT))
        require(frozen.get(relative)==sha256(templates/name),'Worker template is not freeze-bound')
    manifest=read(templates/'manifest.json')
    for name in ('node0.json','node1.json'):
        require(manifest['files'][name]==sha256(templates/name),'Worker template manifest differs')
    protocol_file=ROOT/'protocol/frozen_protocol.json';protocol=validate_protocol(read(protocol_file))
    inputs=read(ROOT/'protocol/public_inputs/inputs_manifest.json')
    require(inputs['frozen'] is True,'Final public inputs required')
    sources={x['case_id']:x for x in read(ROOT/'protocol/author_source_closures.json')['cases']}
    assignment=protocol['lane_assignment'];remote=protocol['remote_experiment_root'];jobs=[]
    for item in inputs['cases']:
        case=item['case_id'];n=int(case.split('_')[0]);arms=['plain_astra','content_agents']
        if n%2==0:arms.reverse()
        for arm in arms:
            run_id='v2_'+case+'_'+arm
            jobs.append({'run_id':run_id,'case_id':case,'arm':arm,'lane_id':assignment[case],
                         'protocol_sha256':sha256(protocol_file),'task_sha256':item['task_sha256'],
                         'input_sha256':item['input_sha256'],'source_sha256':sources[case]['source_sha256'],
                         'input_dir':remote+'/protocol/public_inputs/'+case,
                         'task_file':remote+'/protocol/public_inputs/'+case+'/task.json',
                         'source':remote+'/'+sources[case]['source_root']})
    require(len(jobs)==20,'Exactly20 original assigned attempts required');validate_jobs(protocol,jobs)
    destination.mkdir(mode=0o700,parents=True)
    worker_configs={}
    for host_id in ('node0','node1'):
        cfg=read(Path(templates)/(host_id+'.json'))
        require(cfg['protocol_file']==remote+'/protocol/frozen_protocol.json','Wrong worker protocol')
        path=destination/(host_id+'_worker.json');write_new(path,cfg);path.chmod(0o600)
        worker_configs[host_id]=str(path)
    config={'protocol_file':str(protocol_file),'database':str(destination/'ledger.sqlite'),
            'routes_file':str(destination/'routes.json'),'output':str(destination/'dispatch'),
            'retained_archives':str(destination/'archives'),'teleport_proxy':hosts['teleport_proxy'],
            'hosts':{key:{'hostname':hosts[key],'python':'/usr/bin/python3',
                         'supervisor':remote+'/harness/run_arm.py','retention_supervisor':remote+'/controller/worker_retention.py',
                         'worker_config':remote+'/controller/'+key+'_worker.private.json'}for key in ('node0','node1')}}
    gateway={'upstream_base_url':'https://chatgpt.com/backend-api/codex','auth_file':str(Path.home()/'.codex/auth.json'),
             'database':config['database'],'routes_file':config['routes_file'],'listen_host':'127.0.0.1','listen_port':17862,
             'qualified_product_headers':{'originator':'codex_cli_rs','User-Agent':'codex_cli_rs/0.154.0'}}
    for name,value in [('jobs.json',{'jobs':jobs}),('controller.json',config),('gateway.json',gateway)]:
        p=destination/name;write_new(p,value);p.chmod(0o600)
    return {'jobs':20,'worker_configs':worker_configs,'controller':str(destination/'controller.json')}


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True);p.add_argument('--templates',type=Path,required=True)
    p.add_argument('--hosts',type=Path,required=True);a=p.parse_args();print(json.dumps(build(a.destination,a.templates,read(a.hosts))))
