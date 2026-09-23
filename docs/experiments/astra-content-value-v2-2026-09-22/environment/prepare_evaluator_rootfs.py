"""Create a separate evaluator rootfs from the qualified, unchanged author skeleton."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

DIRECTORIES={'usr','lib','lib64','bin','etc','etc/ssl','etc/ssl/certs','etc/vulkan',
 'etc/alternatives','etc/fonts','proc','dev','tmp','source','work','tools','workflows',
 'input','home','home/agent'}
FILES={'etc/ld.so.cache','etc/os-release','etc/passwd','etc/group'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def inventory(root):
    rows=[]
    for p in sorted(Path(root).rglob('*')):
        rel=str(p.relative_to(root));mode=p.lstat().st_mode
        row={'path':rel,'mode':stat.S_IMODE(mode)}
        if stat.S_ISLNK(mode):row.update(kind='symlink',target=os.readlink(p))
        elif stat.S_ISDIR(mode):row['kind']='directory'
        elif stat.S_ISREG(mode):row.update(kind='file',bytes=p.stat().st_size,sha256=sha(p))
        else:raise ValueError('Special rootfs member: '+rel)
        rows.append(row)
    return rows
def semantic(rows):return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def prepare(source,output,receipt):
    source=Path(source);output=Path(output);receipt=Path(receipt)
    assert source.is_dir() and not source.is_symlink()
    assert not output.exists() and not receipt.exists(),'Create-only targets required'
    before=inventory(source)
    for row in before:
        p=row['path']
        assert ((row['kind']=='directory' and p in DIRECTORIES) or
                (row['kind']=='file' and p in FILES) or
                (row['kind']=='symlink' and p=='bin/sh' and row['target']=='/usr/bin/sh')),p
    assert {r['path'] for r in before if r['kind']=='directory'}==DIRECTORIES
    assert (source/'etc/ld.so.cache').stat().st_size==0
    output.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source,output,symlinks=True)
    for name in ('original','evaluator','evidence'):(output/name).mkdir(mode=0o755)
    (output/'evaluation_driver.py').touch(mode=0o644)
    after=inventory(source);assert after==before,'Author skeleton changed'
    result=inventory(output)
    assert [r for r in result if r['path'] not in ('original','evaluator','evidence','evaluation_driver.py')]==before
    value={'schema_version':'evaluator-rootfs.v2','status':'PASS','source':str(source),
      'output':str(output),'author_skeleton_unchanged':True,'author_layout':before,
      'evaluator_layout':result,'author_layout_sha256':semantic(before),
      'evaluator_layout_sha256':semantic(result),'builder_sha256':sha(__file__),
      'added_targets':['original','evaluator','evidence','evaluation_driver.py'],
      'model_calls':0,'scope':'Empty evaluator mount targets only; author filesystem unchanged.'}
    receipt.parent.mkdir(parents=True,exist_ok=True)
    with receipt.open('x') as f:f.write(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k not in ('author_layout','evaluator_layout','source','output')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--receipt',required=True);a=p.parse_args()
    prepare(a.source,a.output,a.receipt)
