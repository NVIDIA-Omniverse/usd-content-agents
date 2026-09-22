"""Presentation snapshots from frozen evaluator native traces; no author code or animation."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,time
from pathlib import Path

ROOT=Path('/opt/astra-content-value-20260921')
EVAL=ROOT/'evaluations/pilot-v1/01_drawer/plain_astra'
OUT=ROOT/'visuals/pilot-v1/01_drawer/plain_astra'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def prepare():
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdLux
    report=json.loads((EVAL/'report.json').read_text());assert report['pass'] and report['status']=='PASS'
    OUT.mkdir(parents=True,exist_ok=True)
    summary={'case_id':'01_drawer','arm':'plain_astra','status':'PASS','report_sha256':sha(EVAL/'report.json'),'seeds':[{k:x[k] for k in ['seed','status','pass','metrics','checks']} for x in report['trials']],'limits':'Independent bounded drawer-force task with a free0.5kg payload. Contact trace force-labelled field is raw impulse N*s; no hardware or robot-manipulation claim.'}
    dump(OUT/'five_seed_summary.json',summary)
    trial=EVAL/'seed_11';rows=[json.loads(x) for x in (trial/'trace.jsonl').read_text().splitlines()];request=json.loads((trial/'request.json').read_text())
    source=Usd.Stage.Open(str(trial/'scene.usda'));assert source
    source_hash=sha(trial/'scene.usda');trace_hash=sha(trial/'trace.jsonl');frames=[];dependencies={}
    for name,phase,target_time in [('closed_initial','settle',.7),('open_loaded','hold_open',5.1),('closed_returned','hold_closed',10.15)]:
        candidates=[x for x in rows if x['phase']==phase];assert candidates,phase
        row=min(candidates,key=lambda x:abs(x['time_s']-target_time))
        path=OUT/(name+'.usda');source.Flatten().Export(str(path));stage=Usd.Stage.Open(str(path))
        # Package exact external texture bytes beside presentation-only snapshots.
        for prim in stage.Traverse():
            for attr in prim.GetAttributes():
                value=attr.Get()
                if isinstance(value,Sdf.AssetPath) and value.path:
                    original=Path(value.resolvedPath or value.path).resolve()
                    assert original.is_file() and original.is_relative_to(ROOT/'runs/pilot-v1/01_drawer/plain_astra')
                    rel=Path('assets')/original.name;dest=OUT/rel;dest.parent.mkdir(exist_ok=True)
                    digest=sha(original)
                    if dest.exists():assert sha(dest)==digest
                    else:shutil.copy2(original,dest)
                    dependencies[str(rel)]={'source':str(original),'sha256':digest,'bytes':dest.stat().st_size}
                    attr.Set(Sdf.AssetPath(str(rel)))
        for body,key in [(request['drawer_body'],'drawer_pose'),(request['payload_body'],'payload_pose')]:
            prim=stage.GetPrimAtPath(body);scale=Gf.Transform(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).GetScale();xf=UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            for prop in list(prim.GetProperties()):
                if prop.GetName().startswith('xformOp:'):prim.RemoveProperty(prop.GetName())
            xf.SetResetXformStack(True);a=row[key];xf.AddTranslateOp().Set(Gf.Vec3d(*a[:3]));xf.AddOrientOp().Set(Gf.Quatf(a[6],Gf.Vec3f(*a[3:6])));xf.AddScaleOp().Set(Gf.Vec3f(*scale))
        # Color distinguishes the actual free evaluator payload; geometry and pose are unchanged.
        UsdGeom.Gprim(stage.GetPrimAtPath(request['payload_body'])).CreateDisplayColorAttr([Gf.Vec3f(.85,.08,.025)])
        UsdGeom.Xform.Define(stage,'/__ReplayPresentation')
        cam=UsdGeom.Camera.Define(stage,'/__ReplayPresentation/Camera');cam.CreateFocalLengthAttr(42);cam.CreateHorizontalApertureAttr(36);cam.CreateVerticalApertureAttr(27);cam.CreateClippingRangeAttr(Gf.Vec2f(.01,100))
        matrix=Gf.Matrix4d().SetLookAt(Gf.Vec3d(1.6,2.3,2.5),Gf.Vec3d(0,.95,.05),Gf.Vec3d(0,1,0)).GetInverse();UsdGeom.Xformable(cam).AddTransformOp().Set(matrix)
        dome=UsdLux.DomeLight.Define(stage,'/__ReplayPresentation/Dome');dome.CreateIntensityAttr(350)
        keylight=UsdLux.DistantLight.Define(stage,'/__ReplayPresentation/Key');keylight.CreateIntensityAttr(1500);keylight.CreateAngleAttr(20);UsdGeom.Xformable(keylight).AddRotateXYZOp().Set(Gf.Vec3f(-35,-25,0))
        stage.SetStartTimeCode(0);stage.SetEndTimeCode(0);stage.GetRootLayer().Save()
        frames.append({'name':name,'scene':str(path),'scene_sha256':sha(path),'trace_step':row['step'],'time_s':row['time_s'],'phase':row['phase'],'drawer_q_m':row['q_m'],'drawer_pose':row['drawer_pose'],'payload_pose':row['payload_pose'],'output':str(OUT/(name+'.png'))})
    assert sha(trial/'scene.usda')==source_hash and sha(trial/'trace.jsonl')==trace_hash
    config={'prepared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ready_for_gpu':True,'renderer':'OVRTX0.4.1.364340','resolution':[1024,768],'seed':11,'report_sha256':sha(EVAL/'report.json'),'native_trace_sha256':trace_hash,'native_scene_sha256':source_hash,'source_replay_sha256':sha(trial/'replay.usda'),'frames':frames,'dependencies':dependencies,'camera':'/__ReplayPresentation/Camera','provenance':'Each static proof snapshot uses observed native drawer/payload poses from the independently integrated seed11 trace. No submitted animation or author simulation was replayed. Only camera, lights and payload display color are presentation additions. Exact texture bytes are packaged with relative references. The frozen scene, replay, trace, author output and evaluator remain unchanged.','claim_limit':'Images illustrate observed evidence; acceptance derives from all five frozen physical trials, not these renders.'}
    dump(OUT/'replay_render_config.json',config);print(json.dumps(config,indent=2))

def render(gpu_uuid):
    assert gpu_uuid.startswith('GPU-'),'Use a reserved GPU UUID, never a guessed index.'
    config=json.loads((OUT/'replay_render_config.json').read_text());folder=OUT/'renderer_work';folder.mkdir(exist_ok=True);(folder/'.usd-cli').mkdir(exist_ok=True)
    (folder/'.usd-cli/config.toml').write_text('[server]\nallowed_read_roots=["'+str(ROOT)+'"]\nallowed_write_roots=["'+str(OUT)+'"]\n[render]\nrenderer="ovrtx"\novrtx_auto_install=true\n')
    env=os.environ.copy();env.update(WU_OVRTX_VENV_DIR=str(ROOT/'ovrtx-venv'),WU_OVRTX_AUTO_PROVISION='1',CUDA_VISIBLE_DEVICES=gpu_uuid,USD_CLI_SESSION='independent-drawer-proof-'+str(int(time.time())),TMPDIR=str(ROOT/'tmp'),UV_NO_CACHE='1')
    cli=str(ROOT/'repo/.venv/bin/usd-cli');records=[]
    try:
        for frame in config['frames']:
            assert sha(frame['scene'])==frame['scene_sha256'];record={'frame':frame['name'],'gpu_uuid':gpu_uuid,'steps':[]}
            for label,args in [('open',['open',frame['scene']]),('camera',['camera','use',config['camera']]),('render',['render','--renderer','ovrtx','--mode','quality','--res','1024x768','--photoreal','--output',frame['output']])]:
                started=time.time();p=subprocess.run([cli,'--json','--timeout','900']+args,cwd=folder,env=env,capture_output=True,text=True,timeout=930)
                (OUT/(frame['name']+'_'+label+'.json')).write_text(p.stdout);(OUT/(frame['name']+'_'+label+'.stderr')).write_text(p.stderr)
                record['steps'].append({'name':label,'returncode':p.returncode,'started_epoch':started,'duration_s':time.time()-started});assert p.returncode==0,(frame['name'],label,p.returncode)
            from PIL import Image
            import numpy as np
            path=Path(frame['output']);rgb=np.asarray(Image.open(path).convert('RGB'));record['image']={'sha256':sha(path),'bytes':path.stat().st_size,'shape':list(rgb.shape),'std':float(rgb.std()),'unique_rgb':int(len(np.unique(rgb.reshape(-1,3),axis=0)))};records.append(record);dump(OUT/'render_results.json',records)
    finally:
        p=subprocess.run([cli,'--json','server','stop'],cwd=folder,env=env,capture_output=True,text=True,timeout=60);(OUT/'renderer_stop.json').write_text(p.stdout)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--render',action='store_true');p.add_argument('--gpu-uuid');a=p.parse_args()
    if a.prepare:prepare()
    if a.render:render(a.gpu_uuid)
