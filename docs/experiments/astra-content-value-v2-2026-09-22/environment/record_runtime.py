"""Non-model import/provenance preflight; does not simulate, render or accept assets."""
from __future__ import annotations
import argparse, hashlib, importlib, importlib.metadata as metadata, json, os, platform, shutil, subprocess, sys
from pathlib import Path


def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()


def file_tree(root):
 return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix!='.pyc'}


def semantic(receipt):
 return {k:receipt[k] for k in ['code_commit','code_tree','python','architecture','packages','pxr_version','native_resource_files','runtime_lock_files','source_skill_targets','worker_availability','node_packages']}


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root.resolve();repo=root/'repo';fail=[]
 def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
 def check(name,callback):
  try:return callback()
  except Exception as e:fail.append({'check':name,'error':type(e).__name__+': '+str(e)});return None
 packages={d.metadata['Name'].lower().replace('_','-'):d.version for d in metadata.distributions() if d.metadata.get('Name')}
 if 'usd-core' in packages and 'usd-exchange' in packages:fail.append({'check':'single_pxr_owner','error':'usd-core and usd-exchange coexist'})
 modules=['content_agent_workflows.geometry','content_agent_workflows.articulation','content_agent_workflows.physics','content_workflow_cli.cli','geometry_repair.workers','world_understanding.agentic.validation_scaffold','world_understanding.functions.physics.native_behavior_validation','usd_validation_nvidia']
 for module in modules:check(module,lambda module=module:importlib.import_module(module))
 usd=check('pxr.Usd',lambda:importlib.import_module('pxr.Usd'))
 worker={}
 for name in ['SceneOptimizerDeinstanceWorker','GeogramLocalRepairWorker','SdfRebuildWorker','PmpPatchWorker']:
  def probe(name=name):
   cls=getattr(importlib.import_module('geometry_repair.workers'),name);available,reason=cls().available();return {'available':available,'reason':reason}
  worker[name]=check(name,probe)
 for name in ['SceneOptimizerDeinstanceWorker','GeogramLocalRepairWorker']:
  if not worker.get(name) or not worker[name]['available']:fail.append({'check':name,'error':'required repair backend unavailable'})
 for command in ['content-workflow-cli','usd-cli','usd-convert-cad','node','npm','jq','bwrap']:
  if not shutil.which(command):fail.append({'check':command,'error':'executable unavailable'})
 skill_targets={}
 for name in ['content-workflow-geometry','content-workflow-articulation','content-workflow-physics','content-workflow-validation','usd-cli']:
  path=repo/'.agents/skills'/name/'SKILL.md'
  if not path.is_file():fail.append({'check':'skill:'+name,'error':'missing or broken skill'})
  else:skill_targets[name]=sha(path)
 pins=json.loads(Path(__file__).with_name('pins.json').read_text())
 if git('rev-parse','HEAD')!=pins['commit']:fail.append({'check':'code_commit','error':'unexpected pinned source'})
 if git('status','--porcelain','--untracked-files=no'):fail.append({'check':'code_clean','error':'tracked file mutation'})
 locks={r:sha(repo/r) for r in pins['source_files_sha256']}
 node_packages={}
 for name in ['codex','codex-sdk','codex-linux-x64']:
  folder=repo/'agentic/packages/content_workflow_cli/node_modules/@openai'/name
  if not folder.is_dir():fail.append({'check':name,'error':'Missing installed Node package'});continue
  node_packages[name]={'version':json.loads((folder/'package.json').read_text())['version'],'files':file_tree(folder)}
  if not node_packages[name]['version'].startswith('0.154.0'):fail.append({'check':name,'error':'Expected installed0.154.0 dependency override'})
 resources={}
 for rel in ['resources/scene_optimizer_core','resources/geogram']:
  path=root/rel
  if path.exists():resources[rel]=file_tree(path)
  else:fail.append({'check':rel,'error':'resource absent'})
 receipt={'schema_version':'matched-runtime-inventory.v2','code_commit':git('rev-parse','HEAD'),'code_tree':git('rev-parse','HEAD^{tree}'),'python':platform.python_version(),'architecture':platform.machine(),'packages':packages,'pxr_version':list(usd.GetVersion()) if usd else None,'native_resource_files':resources,'runtime_lock_files':locks,'source_skill_targets':skill_targets,'worker_availability':worker,'node_packages':node_packages,'failures':fail,'passed':not fail,'end_to_end_workflow_qualified':False,'limitations':['Import/help/worker availability is not Geometry/Joint/Physics/Validation execution.','Native runtime, nonblank GPU image and synthetic native terminal chain remain required before authoring.']}
 receipt['semantic_sha256']=hashlib.sha256(json.dumps(semantic(receipt),sort_keys=True,separators=(',',':')).encode()).hexdigest()
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps({'passed':receipt['passed'],'failures':fail,'semantic_sha256':receipt['semantic_sha256']}));return 0 if not fail else 2
if __name__=='__main__':raise SystemExit(main())
