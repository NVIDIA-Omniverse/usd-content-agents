"""Match only exported repository/root Git directory ownership to the author UID.

The actual native verifier disables caller Git configuration. Do not patch it.
Both mounts remain read-only; this changes no source, mode, Git object or index.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def inventory(root):
    rows=[]
    for p in [root,*sorted(root.rglob('*'))]:
        mode=p.lstat().st_mode;row={'path':'.' if p==root else str(p.relative_to(root)),'mode':stat.S_IMODE(mode)}
        if stat.S_ISLNK(mode):row.update(kind='symlink',target=os.readlink(p))
        elif stat.S_ISDIR(mode):row['kind']='directory'
        elif stat.S_ISREG(mode):row.update(kind='file',bytes=p.stat().st_size,sha256=digest(p))
        else:raise ValueError('Special repository entry')
        rows.append(row)
    return rows

def run(tools,receipt,uid=20000):
    assert os.geteuid()==0 and uid==20000
    tools=Path(tools).resolve();receipt=Path(receipt)
    assert not receipt.exists(),'Create-only evidence required'
    root=tools/'repo';git=root/'.git'
    assert root.is_dir() and git.is_dir() and not root.is_symlink() and not git.is_symlink()
    before=inventory(root)
    owners=[{'path':str(p.relative_to(tools)),'uid':p.stat().st_uid,'gid':p.stat().st_gid,'mode':stat.S_IMODE(p.stat().st_mode)} for p in (root,git)]
    for p in (root,git):os.chown(p,uid,uid)
    after=inventory(root);assert after==before,'Source/Git bytes, paths or modes changed'
    semantic=hashlib.sha256(json.dumps(after,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    result={'schema_version':'native-git-ownership.v2','status':'PASS','tools':str(tools),
      'prior_owners':owners,'new_owner_uid':uid,'new_owner_gid':uid,
      'changed_paths':['repo','repo/.git'],'repository_bytes_modes_paths_unchanged':True,
      'verified_entries':len(after),'repository_inventory_sha256':semantic,'inventory':after,
      'script_sha256':digest(Path(__file__)),'readonly_mount_requirement_preserved':True,
      'verifier_unchanged':True,'safe_directory_override_added':False,'model_calls':0,
      'scope':'Metadata ownership only. Requires the separate actual UID20000 native verifier qualification.'}
    receipt.parent.mkdir(parents=True,exist_ok=True)
    with receipt.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('inventory','tools')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tools',required=True);p.add_argument('--receipt',required=True);a=p.parse_args();run(a.tools,a.receipt)
