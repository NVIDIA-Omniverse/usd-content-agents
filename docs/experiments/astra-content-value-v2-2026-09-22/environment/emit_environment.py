"""Print shell-safe, arm-neutral environment. No auth is read or written."""
import argparse,hashlib,json,shlex
from pathlib import Path

def environment(root,approved):
 root=root.resolve();exe=root/'resources/geogram/bin/vorpalite'
 actual=hashlib.sha256(exe.read_bytes()).hexdigest()
 if len(approved)!=64 or approved!=actual:raise ValueError('Geogram executable is not independently approved at this digest')
 return {'PATH':str(root/'repo/agentic/packages/content_workflow_cli/node_modules/.bin')+':'+str(root/'repo/.venv/bin')+':'+str(root/'resources/geogram/bin')+':'+str(root/'tools/uv')+':/usr/local/bin:/usr/bin:/bin','LD_LIBRARY_PATH':str(root/'resources/geogram/lib'),'WU_SO_PACKAGE_DIR':str(root/'resources/scene_optimizer_core'),'WU_SO_PYTHON':str(root/'repo/.venv/bin/python'),'WU_OVRTX_VENV_DIR':str(root/'ovrtx-venv'),'WU_OVPHYSX_VENV_DIR':str(root/'ovphysx-venv'),'WU_OVRTX_AUTO_PROVISION':'1','WU_OVPHYSX_AUTO_PROVISION':'1','GEOMETRY_REPAIR_GEOGRAM_EXECUTABLE_SHA256':approved,'UV_NO_CACHE':'1','UV_PYTHON_INSTALL_DIR':str(root/'tools/python'),'TMPDIR':str(root/'tmp'),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--approved-geogram-sha256',required=True);a=p.parse_args()
 for k,v in environment(a.root,a.approved_geogram_sha256).items():print('export '+k+'='+shlex.quote(v))
