"""Provision exact repository runtime locks; constructor/import probes, no scenes or steps."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from bootstrap import guard

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();repo=root/'repo'
os.environ.update(USD_CLI_UV_EXECUTABLE=str(root/'tools/uv/uv'),UV_NO_CACHE='1',TMPDIR=str(root/'tmp'),WU_OVPHYSX_VENV_DIR=str(root/'ovphysx-venv'),WU_OVRTX_VENV_DIR=str(root/'ovrtx-venv'),WU_OVPHYSX_AUTO_PROVISION='1',WU_OVRTX_AUTO_PROVISION='1')
# Each backend owns its isolated exact-lock venv and readiness marker.
from usd_core.physics_runtime import _ovphysx_python
from usd_core.render.ovrtx import _provision_venv,_venv_matches_pin
start=time.time();before=guard(root);physics=_ovphysx_python(root/'ovphysx-venv');guard(root);_provision_venv(root/'ovrtx-venv');after=guard(root)
versions={}
for name in ['ovphysx-venv','ovrtx-venv']:
 script='import importlib.metadata as m,json; print(json.dumps({d.metadata["Name"]:d.version for d in m.distributions()}))'
 versions[name]=json.loads(subprocess.check_output([str(root/name/'bin/python'),'-I','-c',script],text=True))
receipt={'status':'EXACT_RUNTIME_LOCKS_INSTALLED','runtime_profiles':versions,'ovrtx_lock_probe_pass':_venv_matches_pin(root/'ovrtx-venv'),'elapsed_s':time.time()-start,'opt_bytes_before':before,'opt_bytes_after':after,'lock_hashes':{str(f.relative_to(repo)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [repo/'apps/usd_cli/src/usd_core/pylock.ovphysx-runtime.toml',repo/'apps/usd_cli/src/usd_core/render/pylock.ovrtx-runtime.toml']},'models_launched':False,'scene_loaded':False,'simulation_steps':0,'rendered_frames':0,'remaining':'Actual per-GPU visible render and native synthetic task chain are not established by package installation.'}
(root/'environment/evidence/native_provision.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
