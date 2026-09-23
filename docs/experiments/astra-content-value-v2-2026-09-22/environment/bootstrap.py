"""Install one matched environment. No model, authoring, solver, or render jobs."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, urllib.request, zipfile, time
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def run(args, *, cwd=None, env=None):
    print(json.dumps({'command':args,'cwd':str(cwd) if cwd else None}),flush=True)
    subprocess.run(args,cwd=cwd,env=env,check=True)


def guard(root, limit=23*1024**3):
    used=int(subprocess.check_output(['du','-sx','-B1','/opt'],text=True).split()[0])
    if used > limit:raise RuntimeError('Ephemeral /opt usage exceeds23GiB stop threshold')
    return used


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--constraints',type=Path);p.add_argument('--skip-python',action='store_true');a=p.parse_args()
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    pins=json.loads(Path(__file__).with_name('pins.json').read_text());repo=root/'repo'; evidence=root/'environment/evidence';evidence.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(UV_NO_CACHE='1',PIP_NO_CACHE_DIR='1',TMPDIR=str(root/'tmp'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    (root/'tmp').mkdir(exist_ok=True)
    if a.constraints:
        env['UV_CONSTRAINT']=str(a.constraints.resolve())
    guard(root)
    if not repo.exists():
        run(['git','init',str(repo)])
        run(['git','remote','add','origin',pins['repository']],cwd=repo)
        run(['git','fetch','--depth=1','origin',pins['commit']],cwd=repo)
        run(['git','checkout','--detach','FETCH_HEAD'],cwd=repo)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if head!=pins['commit']:raise RuntimeError('Existing checkout has unexpected commit')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo):raise RuntimeError('Tracked source changed')
    for name,expected in pins['source_files_sha256'].items():
        if digest(repo/name)!=expected:raise RuntimeError('Pinned setup/lock file changed: '+name)
    resource=root/'resources/scene_optimizer_core';so=pins['scene_optimizer']
    if not resource.exists():
        archive=root/'tmp/scene_optimizer.zip'
        if not archive.exists() or digest(archive)!=so['sha256']:
            urllib.request.urlretrieve(so['url'],archive)
        if archive.stat().st_size!=so['bytes'] or digest(archive)!=so['sha256']:raise RuntimeError('SceneOptimizer archive hash mismatch')
        with zipfile.ZipFile(archive) as z:
            for n in z.namelist():
                if Path(n).is_absolute() or '..' in Path(n).parts:raise RuntimeError('Unsafe archive path')
        resource.mkdir(parents=True)
        run(['unzip','-q',str(archive),'-d',str(resource)])
        for n in ['python','lib','extraLibs','usdpy']:
            if not (resource/n).is_dir():raise RuntimeError('SceneOptimizer layout missing '+n)
        (resource/'.so_core_platform').write_text('platform=manylinux_2_35_x86_64\nurl_sha256='+hashlib.sha256(so['url'].encode()).hexdigest()+'\n')
        (evidence/'scene_optimizer_archive.json').write_text(json.dumps(so,indent=2)+'\n')
        archive.unlink()
    env['SO_CORE_BUILD_RESOURCES']=str(root/'resources')
    env['WU_SO_PACKAGE_DIR']=str(resource)
    env['WU_SO_PYTHON']=str(repo/'.venv/bin/python')
    if not a.skip_python:
        run(['bash','scripts/setup_content_agent.sh'],cwd=repo,env=env)
        guard(root)
        run(['uv','pip','install','--python',str(repo/'.venv/bin/python'),'--overrides','apps/usd_cli/requirements/usd-exchange-override.txt','usd-convert-cad==0.2.0','cadquery-ocp==7.8.1.1','vtk==9.3.1','rtree','pytest','pytest-asyncio'],cwd=repo,env=env)
        frozen=subprocess.check_output(['uv','pip','freeze','--python',str(repo/'.venv/bin/python'),'--exclude-editable'],cwd=repo,env=env,text=True)
        (evidence/'thirdparty.constraints.txt').write_text(frozen)
        run(['uv','pip','check','--python',str(repo/'.venv/bin/python')],cwd=repo,env=env)
        run(['bash','scripts/sync_agent_skills.sh','--check'],cwd=repo,env=env)
        run(['npm','install','--prefix',str(repo/'agentic/packages/content_workflow_cli'),'--no-save','--package-lock=false','@openai/codex-sdk@0.154.0','@openai/codex@0.154.0'],cwd=repo,env=env)
    receipt={'status':'SETUP_COMPLETED_NOT_RUNTIME_QUALIFIED','commit':head,'tree':subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=repo,text=True).strip(),'source_unchanged':not bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo)),'opt_bytes':guard(root),'constraints_input_sha256':digest(a.constraints) if a.constraints else None,'models_launched':False,'physics_or_render_launched':False,'remaining':['Geogram independently approved build','Native runtime provisioning/probes','Actual nonblank OVRTX image per isolated GPU','Synthetic native Geometry/Joint/Physics/Validation closure','Cross-host semantic inventory equality']}
    (evidence/'bootstrap.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
