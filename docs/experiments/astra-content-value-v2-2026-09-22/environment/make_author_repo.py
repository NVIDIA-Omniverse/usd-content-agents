"""Create a real Git source-only author view with no retained outcomes/history.
Supervisor keeps the original pinned clone and verifies all retained file bytes.
This derived view must be identical in both arms and mounted at the installed
editable-code path; no package source may point outside the sandbox's view.
"""
import argparse,hashlib,json,os,subprocess,tarfile,io,posixpath
from pathlib import Path

EXCLUDE_PREFIXES=('tests/','docs/experiments/')
EXCLUDE_EXACT={'agentic/docs/drawer_geometry_capstone.md'}
def excluded(path):return path in EXCLUDE_EXACT or path.startswith(EXCLUDE_PREFIXES)
def safe_member(member):
 path=member.name
 if path.startswith('/') or '..' in Path(path).parts:return False
 if member.issym():
  target=posixpath.normpath(posixpath.join(posixpath.dirname(path),member.linkname))
  return not member.linkname.startswith('/') and target!='..' and not target.startswith('../')
 return member.isdir() or member.isfile()
def main():
 p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();repo=a.repository.resolve();out=a.output.resolve()
 pins=json.loads(Path(__file__).with_name('pins.json').read_text());commit=pins['commit']
 if out.exists():raise ValueError('Create-only author snapshot')
 if subprocess.check_output(['git','-C',str(repo),'rev-parse',commit],text=True).strip()!=commit:raise ValueError('Wrong source commit')
 data=subprocess.check_output(['git','-C',str(repo),'archive',commit]);out.mkdir(parents=True);included=[];removed=[]
 with tarfile.open(fileobj=io.BytesIO(data)) as archive:
  for member in archive.getmembers():
   if excluded(member.name):removed.append(member.name);continue
   if not safe_member(member):raise ValueError('Unsafe Git archive entry '+member.name)
   archive.extract(member,out)
   if member.isfile():included.append({'path':member.name,'sha256':hashlib.sha256((out/member.name).read_bytes()).hexdigest()})
 env=os.environ.copy();env.update(GIT_AUTHOR_NAME='Controlled experiment source snapshot',GIT_AUTHOR_EMAIL='snapshot@example.invalid',GIT_COMMITTER_NAME='Controlled experiment source snapshot',GIT_COMMITTER_EMAIL='snapshot@example.invalid',GIT_AUTHOR_DATE='2026-09-22T00:00:00+00:00',GIT_COMMITTER_DATE='2026-09-22T00:00:00+00:00')
 for args in [['init'],['-c','core.autocrlf=false','add','--force','--all'],['-c','commit.gpgsign=false','commit','-m','Source-only experiment implementation from '+commit]]:subprocess.run(['git','-C',str(out),*args],check=True,env=env,stdout=subprocess.DEVNULL)
 receipt={'schema_version':'source-only-author-git-view.v2','upstream_commit':commit,'upstream_tree':pins['tree'],'author_snapshot_commit':subprocess.check_output(['git','-C',str(out),'rev-parse','HEAD'],text=True).strip(),'included_file_hashes':included,'excluded_paths':removed,'history_copied':False,'exclusion_scope':'Supervisor tests, public experiment outcomes and case-specific capstone diary; package implementation and root skill surfaces unchanged.','requires_both_arms_identical':True,'not_claimed_upstream_commit':True}
 a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ['included_file_hashes','excluded_paths']}))
if __name__=='__main__':main()
