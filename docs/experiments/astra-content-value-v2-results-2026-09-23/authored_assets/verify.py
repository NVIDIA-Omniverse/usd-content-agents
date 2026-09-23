"""Verify this exact bundle; never execute submitted code or compose a stage."""
import argparse,hashlib,json
from pathlib import Path,PurePosixPath

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def verify(root,usd_review=False):
 root=Path(root).resolve();m=json.loads((root/'manifest.json').read_text());expected={}
 for r in m['files']:
  p=PurePosixPath(r['path']);assert not p.is_absolute() and '..' not in p.parts and str(p) not in expected
  expected[str(p)]=r
 actual=set()
 for p in root.rglob('*'):
  assert not p.is_symlink(), 'Symlink in bundle'
  if p.is_file():actual.add(p.relative_to(root).as_posix())
 assert actual==set(expected)|{'manifest.json'}, 'Unexpected or missing bundle members'
 for name,r in expected.items():
  p=root/name;assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256'], 'File identity mismatch: '+name
  if r['role']=='exact_submitted_asset_or_dependency':assert r['sha256']==r['source_identity']['original_sha256'] and r['bytes']==r['source_identity']['original_bytes']
 if usd_review:
  from pxr import Sdf,Usd
  assert Usd.GetVersion()==(0,25,5), 'Decoded-text proof was produced with USD25.5'
  for run in m['runs']:
   for v in run['privacy_review']:
    if not v['path'].endswith('.usd'):continue
    l=Sdf.Layer.OpenAsAnonymous(str(root/v['path']));text=l.ExportToString();assert hashlib.sha256(text.encode()).hexdigest()==v['decoded_text_sha256']
    for d in v['bounded_locator_dispositions']:
     assert d['field']=='rootLayer.documentation' and hashlib.sha256(l.documentation.encode()).hexdigest()==d['value_sha256']
 print(json.dumps({'membership_verified':True,'verified_files':len(expected)+1,'payload_files':m['submitted_payload_files'],'payload_bytes':m['submitted_payload_bytes'],'usd_decoded_identity_rechecked':usd_review},sort_keys=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',nargs='?',default='.');p.add_argument('--usd-review',action='store_true');a=p.parse_args();verify(a.root,a.usd_review)
