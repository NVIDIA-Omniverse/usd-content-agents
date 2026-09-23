"""Supervisor-side provider-free test launcher; no author or evaluator mounted."""
import argparse,json,os,shutil,subprocess,resource
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--tools',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);out.chmod(0o777)
 for file in ['namespace_smoke.py','smoke_backends.py']:shutil.copy2(Path(__file__).with_name(file),out/file)
 env=json.loads((a.tools/'environment.json').read_text())
 home=out/'home';home.mkdir();home.chmod(0o777)
 locks=[]
 for name in ['ovphysx-venv.provision.lock','ovrtx-venv.provision.lock']:
  path=out/name;path.write_bytes(b'');path.chmod(0o600);subprocess.run(['sudo','chown','65534:65534',str(path)],check=True);locks+=['--bind',str(path),'/tools/'+name]
 cmd=['sudo','bwrap','--unshare-pid','--unshare-net','--unshare-ipc','--unshare-uts','--new-session','--die-with-parent','--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64','--symlink','usr/bin','/bin','--symlink','usr/sbin','/sbin','--dir','/etc','--ro-bind','/etc/ld.so.cache','/etc/ld.so.cache','--proc','/proc','--dev','/dev','--perms','1777','--tmpfs','/dev/shm','--perms','1777','--tmpfs','/tmp','--dir','/home','--bind',str(home),'/home/agent','--ro-bind',str(a.tools.resolve()),'/tools','--bind',str(out),'/work','--chdir','/work','--clearenv','--setenv','HOME','/home/agent']
 for k,v in env.items():cmd+=['--setenv',k,v]
 cmd+=locks
 cmd+=['--','/usr/bin/setpriv','--reuid','65534','--regid','65534','--clear-groups','--bounding-set=-all','--inh-caps=-all','--ambient-caps=-all','--no-new-privs','--','/tools/main-venv/bin/python','/work/namespace_smoke.py']
 (out/'launch.json').write_text(json.dumps({'command':cmd,'scope':'Read-only common tools, empty network, no model/auth/evaluator/source mounts'},indent=2)+'\n')
 with (out/'execution.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=360,preexec_fn=lambda:resource.setrlimit(resource.RLIMIT_CORE,(0,0)))
 print(json.dumps({'returncode':r.returncode,'output':str(out)}));return r.returncode
if __name__=='__main__':raise SystemExit(main())
