"""Create-only /tools export. Never mutate linked originals or scored files.

Only installation-generated locators are rewritten. Source-only Git retains the
same implementation bytes for both arms, with no supervisor execution history.
The export is NOT qualified until the actual read-only namespace smoke passes.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, time, shlex
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def atomic_bytes(path,data):
 mode=path.stat().st_mode & 0o777
 temp=path.with_name(path.name+'.relocating')
 if temp.exists():raise ValueError('Unexpected relocation temporary file')
 temp.write_bytes(data);temp.chmod(mode);os.replace(temp,path)

def copy_tree(source,target):
 def ignore(folder,names):
  return [n for n in names if n=='__pycache__' or n.endswith(('.pyc','.pyo')) or n=='.lock']
 shutil.copytree(source,target,symlinks=True,copy_function=os.link,ignore=ignore)
 # Staging venv roots can be0700. Export directories are separate inodes.
 for directory in [target,*[p for p in target.rglob('*') if p.is_dir() and not p.is_symlink()]]:
  directory.chmod(directory.stat().st_mode | 0o555)

def relocate_tree(target,mappings):
 changes=[]
 for p in sorted(target.rglob('*')):
  if p.is_symlink():
   before=os.readlink(p);after=before
   for old,new in mappings:after=after.replace(old,new)
   if after!=before:
    p.unlink();p.symlink_to(after);changes.append({'path':str(p.relative_to(target)),'kind':'symlink','before':before,'after':after})
   continue
  if not p.is_file():continue
  # Explicitly bounded installation-generated metadata/entrypoints, not package code.
  generated=(p.name=='pyvenv.cfg' or p.suffix=='.pth' or p.name=='direct_url.json' or p.name.startswith('_sysconfigdata_') or p.name.startswith('_sysconfig_vars_') or
   p.name.startswith('__editable__') or p.name.startswith('.usd-cli-') or
   p.parent.name=='bin' and p.stat().st_size<1024*1024)
  if not generated:continue
  data=p.read_bytes()
  try:data.decode('utf-8')
  except UnicodeDecodeError:continue
  changed=data
  for old,new in mappings:changed=changed.replace(old.encode(),new.encode())
  # Python's historical venv creation command can retain a random staging path;
  # it is not used by runtime selection. Keep interpreter/home/version entries.
  if p.name=='pyvenv.cfg':changed=b'\n'.join(line for line in changed.splitlines() if not line.startswith(b'command = '))+b'\n'
  if changed!=data:
   oldsha=hashlib.sha256(data).hexdigest();atomic_bytes(p,changed)
   changes.append({'path':str(p.relative_to(target)),'kind':'generated_locator','before_sha256':oldsha,'after_sha256':sha(p)})
 return changes

def manifest(root):
 result=[]
 for p in sorted(root.rglob('*')):
  if p.is_symlink():result.append({'path':str(p.relative_to(root)),'symlink':os.readlink(p)})
  elif p.is_file():result.append({'path':str(p.relative_to(root)),'sha256':sha(p),'bytes':p.stat().st_size})
 return result

def staging_mapping(root):
 """Read the installer's recorded staging target, never guess random suffixes."""
 cfg=(root/'ovrtx-venv/pyvenv.cfg').read_text()
 commands=[line.partition(' = ')[2] for line in cfg.splitlines() if line.startswith('command = ')]
 if not commands:return []
 command=shlex.split(commands[0]);path=Path(command[-1])
 if path.parent==root and path.name.startswith('.ovrtx-venv.staging-'):return [(str(path),'/tools/ovrtx-venv')]
 if path==root/'ovrtx-venv':return []
 raise ValueError('Unrecognized recorded OVRTX staging target')

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();out=a.output.resolve();repo=root/'repo';start=time.time()
 if out.exists() or a.receipt.exists():raise ValueError('Export and receipt are create-only')
 if not out.is_relative_to(root):raise ValueError('Export must remain inside experiment root')
 out.mkdir(parents=True)
 python=Path(subprocess.check_output([str(repo/'.venv/bin/python'),'-c','import sys;print(sys.base_prefix)'],text=True).strip())
 if not python.is_relative_to(root/'tools/python'):raise ValueError('Unexpected managed Python base')
 source_receipt=a.receipt.with_name(a.receipt.stem+'_source_git.json')
 subprocess.run([sys.executable,str(Path(__file__).with_name('make_author_repo.py')),'--repository',str(repo),'--output',str(out/'repo'),'--receipt',str(source_receipt)],check=True)
 pairs=[(python,out/'python'),(repo/'.venv',out/'main-venv'),(root/'ovphysx-venv',out/'ovphysx-venv'),(root/'ovrtx-venv',out/'ovrtx-venv'),(root/'resources/scene_optimizer_core',out/'resources/scene_optimizer_core'),(root/'resources/geogram',out/'resources/geogram'),(repo/'agentic/packages/content_workflow_cli/node_modules',out/'repo/agentic/packages/content_workflow_cli/node_modules')]
 for src,dst in pairs:dst.parent.mkdir(parents=True,exist_ok=True);copy_tree(src,dst)
 mappings=staging_mapping(root)+[(str(repo/'.venv'),'/tools/main-venv'),(str(python),'/tools/python'),(str(root/'ovphysx-venv'),'/tools/ovphysx-venv'),(str(root/'ovrtx-venv'),'/tools/ovrtx-venv'),(str(repo),'/tools/repo'),(str(root/'resources'),'/tools/resources')]
 changes=[]
 for name in ['python','main-venv','ovphysx-venv','ovrtx-venv']:
  for item in relocate_tree(out/name,mappings):item['path']=name+'/'+item['path'];changes.append(item)
 # Venv launch symlinks must always select the exported interpreter.
 for name in ['main-venv','ovphysx-venv','ovrtx-venv']:
  for basename in ['python','python3','python3.12']:
   path=out/name/'bin'/basename
   if path.is_symlink():path.unlink()
   elif path.exists():raise ValueError('Unexpected material interpreter file')
   path.symlink_to('/tools/python/bin/python3.12')
 binpath=out/'bin';binpath.mkdir()
 for exe in ['python','python3','content-workflow-cli','usd-cli','usd-convert-cad','pytest']:
  if not (out/'main-venv/bin'/exe).exists() and not (out/'main-venv/bin'/exe).is_symlink():raise ValueError('Missing entrypoint '+exe)
  (binpath/exe).symlink_to('/tools/main-venv/bin/'+exe)
 os.link(root/'tools/uv/uv',binpath/'uv');os.link(root/'tools/uv/uvx',binpath/'uvx')
 (binpath/'codex').write_text('#!/bin/sh\nexec /usr/bin/node /tools/repo/agentic/packages/content_workflow_cli/node_modules/@openai/codex/bin/codex.js "$@"\n');(binpath/'codex').chmod(0o755)
 (binpath/'vorpalite').symlink_to('/tools/resources/geogram/bin/vorpalite')
 # Native daemons take these locks even when provisioning is disabled. The
 # harness overlays only these two placeholders with lane-private writable
 # files; all executable/runtime contents remain on the read-only mount.
 for name in ['ovphysx-venv.provision.lock','ovrtx-venv.provision.lock']:(out/name).write_bytes(b'')
 geosha=sha(out/'resources/geogram/bin/vorpalite')
 env={'PATH':'/tools/bin:/tools/main-venv/bin:/usr/local/bin:/usr/bin:/bin','LD_LIBRARY_PATH':'/tools/python/lib:/tools/resources/geogram/lib','WU_SO_PACKAGE_DIR':'/tools/resources/scene_optimizer_core','WU_SO_PYTHON':'/tools/main-venv/bin/python','WU_OVPHYSX_VENV_DIR':'/tools/ovphysx-venv','WU_OVRTX_VENV_DIR':'/tools/ovrtx-venv','WU_OVPHYSX_AUTO_PROVISION':'0','WU_OVRTX_AUTO_PROVISION':'0','GEOMETRY_REPAIR_GEOGRAM_EXECUTABLE_SHA256':geosha,'UV_NO_CACHE':'1','PIP_NO_CACHE_DIR':'1','PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','TMPDIR':'/tmp','GIT_CONFIG_COUNT':'1','GIT_CONFIG_KEY_0':'safe.directory','GIT_CONFIG_VALUE_0':'/tools/repo'}
 (out/'environment.json').write_text(json.dumps(env,indent=2,sort_keys=True)+'\n')
 # Verify retained source bytes and all relocation originals after hardlink breaks.
 source=json.loads(source_receipt.read_text());checks={}
 checks['source_files_unchanged']=all(sha(out/'repo'/f['path'])==f['sha256'] for f in source['included_file_hashes'])
 checks['source_git_clean']=not subprocess.check_output(['git','-C',str(out/'repo'),'status','--porcelain','--untracked-files=no'],text=True).strip()
 originals=[]
 for c in changes:
  if c['kind']!='generated_locator':continue
  rel=Path(c['path']);name=rel.parts[0];base=dict((str(d.relative_to(out)),s) for s,d in pairs)[name];original=base/Path(*rel.parts[1:])
  originals.append({'path':str(original),'sha256':sha(original),'unchanged':sha(original)==c['before_sha256']})
 checks['all_modified_hardlink_originals_unchanged']=all(x['unchanged'] for x in originals)
 checks['auto_provision_disabled']=all(env[k]=='0' for k in ['WU_OVPHYSX_AUTO_PROVISION','WU_OVRTX_AUTO_PROVISION'])
 files=manifest(out)
 receipt={'schema_version':'namespace-tools-export.v2','passed':all(checks.values()),'checks':checks,'elapsed_seconds':time.time()-start,'source_git_receipt_sha256':sha(source_receipt),'source_git_commit':source['author_snapshot_commit'],'upstream_commit':source['upstream_commit'],'generated_relocations':changes,'originals_unchanged':originals,'files':files,'manifest_sha256':hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'qualification':'NOT_YET_NAMESPACE_QUALIFIED','scope':'Identical tools/code for both arms; workflow activation/non-use is audited, not a source read ban. Read-only mount required. No author/evaluator/history/auth records included.'}
 a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ['files','generated_relocations','originals_unchanged']}))
 if not receipt['passed']:raise SystemExit(2)
if __name__=='__main__':main()
