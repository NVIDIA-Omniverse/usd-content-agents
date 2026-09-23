"""Reuse this experiment's successfully tested isolated launcher configuration."""
import argparse,hashlib,json,os,resource,shutil,subprocess
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--qualified-smoke',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prior=a.qualified_smoke.resolve();out=a.output.resolve()
 if not json.loads((prior/'receipt.json').read_text())['passed']:raise ValueError('Require actual passing namespace smoke')
 old=json.loads((prior/'launch.json').read_text())['command']
 if old[:2]!=['sudo','bwrap'] or old[-2:]!=['/tools/main-venv/bin/python','/work/namespace_smoke.py']:raise ValueError('Unexpected qualified command')
 out.mkdir(parents=True,exist_ok=False);out.chmod(0o777);(out/'home').mkdir(mode=0o777);(out/'home').chmod(0o777);shutil.copy2(Path(__file__).with_name('namespace_native_step.py'),out/'namespace_native_step.py')
 for name in ['ovphysx-venv.provision.lock','ovrtx-venv.provision.lock']:
  path=out/name;path.write_bytes(b'');path.chmod(0o600);subprocess.run(['sudo','chown','65534:65534',str(path)],check=True)
 cmd=[str(out)+x[len(str(prior)):] if x==str(prior) or x.startswith(str(prior)+'/') else x for x in old];cmd[-1]='/work/namespace_native_step.py'
 (out/'launch.json').write_text(json.dumps({'command':cmd,'qualified_smoke_receipt_sha256':hashlib.sha256((prior/'receipt.json').read_bytes()).hexdigest(),'qualified_smoke_launch_sha256':hashlib.sha256((prior/'launch.json').read_bytes()).hexdigest(),'test_source_sha256':hashlib.sha256((out/'namespace_native_step.py').read_bytes()).hexdigest()},indent=2)+'\n')
 with (out/'execution.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180,preexec_fn=lambda:resource.setrlimit(resource.RLIMIT_CORE,(0,0)))
 print(json.dumps({'returncode':r.returncode,'output':str(out)}));return r.returncode
if __name__=='__main__':raise SystemExit(main())
