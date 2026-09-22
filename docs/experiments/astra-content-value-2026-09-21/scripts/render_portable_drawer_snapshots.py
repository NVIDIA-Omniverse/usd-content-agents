"""Render exact packaged independent-trace snapshots. No simulation or author code runs.

Example: python render_portable_drawer_snapshots.py --snapshots /path/to/plain_astra
  --usd-cli /path/to/runtime/.venv/bin/usd-cli --ovrtx-venv /path/to/ovrtx-venv
  --gpu-uuid GPU-... . Snapshot geometry/assets must accompany replay_render_config.json.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,time
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
p=argparse.ArgumentParser();p.add_argument('--snapshots',type=Path,required=True);p.add_argument('--usd-cli',type=Path,required=True);p.add_argument('--ovrtx-venv',type=Path,required=True);p.add_argument('--gpu-uuid',required=True);a=p.parse_args()
assert a.gpu_uuid.startswith('GPU-');out=a.snapshots.resolve();cli=a.usd_cli.resolve();venv=a.ovrtx_venv.resolve()
assert cli.is_file() and venv.is_dir();config=json.loads((out/'replay_render_config.json').read_text());assert len(config['frames'])==3
assert not (out/'render_provenance.json').exists(),'Preserve an existing render attempt; use another snapshot directory for a retry.'
for f in config['frames']:assert sha(out/Path(f['scene']).name)==f['scene_sha256']
for name,data in config['dependencies'].items():
    path=(out/name).resolve();assert path.is_relative_to(out) and sha(path)==data['sha256']
work=out/'renderer_work';work.mkdir(exist_ok=True);(work/'.usd-cli').mkdir(exist_ok=True)
(work/'.usd-cli/config.toml').write_text('[server]\nallowed_roots='+json.dumps([str(out)])+'\nallowed_read_roots='+json.dumps([str(out)])+'\nallowed_write_roots='+json.dumps([str(out)])+'\n[render]\nrenderer="ovrtx"\novrtx_auto_install=true\n')
session='independent-drawer-proof-'+str(int(time.time()));env=os.environ.copy();env.update(WU_OVRTX_VENV_DIR=str(venv),WU_OVRTX_AUTO_PROVISION='1',CUDA_VISIBLE_DEVICES=a.gpu_uuid,USD_CLI_SESSION=session,UV_NO_CACHE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
tmp=work/'tmp';tmp.mkdir(exist_ok=True);env['TMPDIR']=str(tmp)
started=time.time();provenance={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'renderer_requested':'OVRTX','qualification_version':'0.4.1.364340','gpu_uuid':a.gpu_uuid,'cuda_visible_devices':a.gpu_uuid,'usd_cli_session':session,'usd_cli':str(cli),'ovrtx_venv':str(venv),'snapshot_config_sha256':sha(out/'replay_render_config.json'),'script_sha256':sha(Path(__file__)),'resolution':[1024,768],'mode':'quality','source':'Packaged source-faithful static snapshots with actual independent native-trace poses; no author animation or physics code executed.','frames':[]}
dump(out/'render_provenance.json',provenance)
try:
    for f in config['frames']:
        name=f['name'];scene=out/Path(f['scene']).name;output=out/(name+'.png');assert not output.exists()
        rec={'frame':name,'snapshot_sha256':sha(scene),'gpu_uuid':a.gpu_uuid,'steps':[]}
        for label,args in [('open',['open',str(scene)]),('camera',['camera','use',config['camera']]),('render',['render','--renderer','ovrtx','--mode','quality','--res','1024x768','--photoreal','--output',str(output)])]:
            t=time.time();r=subprocess.run([str(cli),'--json','--timeout','900']+args,cwd=work,env=env,capture_output=True,text=True,timeout=930)
            (out/(name+'_'+label+'.json')).write_text(r.stdout);(out/(name+'_'+label+'.stderr')).write_text(r.stderr)
            rec['steps'].append({'name':label,'returncode':r.returncode,'started_epoch':t,'duration_s':time.time()-t});provenance['active_frame']=rec;dump(out/'render_provenance.json',provenance);assert r.returncode==0,(name,label,r.returncode)
        from PIL import Image
        import numpy as np
        pixels=np.asarray(Image.open(output).convert('RGB'));rec['image']={'sha256':sha(output),'bytes':output.stat().st_size,'shape':list(pixels.shape),'std':float(pixels.std()),'unique_rgb':int(len(np.unique(pixels.reshape(-1,3),axis=0)))}
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
        rec['allocated_gpu_processes_after_render']=[line for line in gpu.stdout.splitlines() if a.gpu_uuid in line];rec['gpu_query_returncode']=gpu.returncode
        provenance['frames'].append(rec);dump(out/'render_results.json',provenance['frames']);dump(out/'render_provenance.json',provenance)
finally:
    r=subprocess.run([str(cli),'--json','server','stop'],cwd=work,env=env,capture_output=True,text=True,timeout=60)
    (out/'renderer_stop.json').write_text(r.stdout);(out/'renderer_stop.stderr').write_text(r.stderr)
    provenance.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.time()-started,server_stop_returncode=r.returncode,completed_frames=len(provenance['frames']))
    dump(out/'render_provenance.json',provenance)
print(json.dumps(provenance,indent=2))
