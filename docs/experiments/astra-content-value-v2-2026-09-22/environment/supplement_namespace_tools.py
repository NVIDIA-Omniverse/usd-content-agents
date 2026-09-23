"""Add only native lock placeholders and the exact authorized model catalog.
No installed source/runtime byte is changed. Existing parent receipt is retained.
Catalog contents are deliberately never printed or included in this receipt.
"""
import argparse,hashlib,json
from pathlib import Path
from export_namespace_tools import sha,manifest
CATALOG_SHA256='e612a56f9fef06041483ddd20cbea586f50d2fbf12e558e852dfa40bd332f976'
def main():
 p=argparse.ArgumentParser();p.add_argument('--tools',type=Path,required=True);p.add_argument('--parent-receipt',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();root=a.tools.resolve();parent=json.loads(a.parent_receipt.read_text())
 if a.receipt.exists():raise ValueError('Supplement receipt is create-only')
 if not parent['passed'] or sha(a.catalog)!=CATALOG_SHA256:raise ValueError('Parent export or exact catalog hash invalid')
 before={f['path']:f for f in manifest(root)};expected={f['path']:f for f in parent['files']}
 # Git administrative timestamps may refresh; installed and tracked bytes may not.
 mismatched=[name for name,value in expected.items() if not name.startswith('repo/.git/') and before.get(name)!=value]
 if mismatched:raise ValueError('Export changed since parent receipt: '+repr(mismatched[:10]))
 wanted=[('ovphysx-venv.provision.lock',b''),('ovrtx-venv.provision.lock',b''),('astra_model_catalog.json',a.catalog.read_bytes())]
 unexpected=[name for name in before if name not in expected and not name.startswith('repo/.git/') and name not in dict(wanted)]
 if unexpected:raise ValueError('Unexpected files in export: '+repr(unexpected[:10]))
 for name,content in wanted:
  path=root/name
  if path.is_symlink() or path.exists() and path.read_bytes()!=content:raise ValueError('Conflicting existing supplement '+name)
 additions=[]
 for name,content in wanted:
  path=root/name
  if path.exists():
   if path.is_symlink() or path.read_bytes()!=content:raise ValueError('Conflicting existing supplement '+name)
  else:path.write_bytes(content);path.chmod(0o644);additions.append(name)
 files=manifest(root);result=dict(parent);result.update(schema_version='namespace-tools-export.v2',parent_receipt_sha256=sha(a.parent_receipt),supplement_script_sha256=sha(Path(__file__)),added_files=additions,model_catalog_sha256=CATALOG_SHA256,files=files,manifest_sha256=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),supplement_scope='Only empty lane-private lock mount targets and exact read-only single-Astra catalog; catalog content not reproduced in receipt. Requalify after mount configuration changes.')
 a.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'passed':True,'added_files':additions,'manifest_sha256':result['manifest_sha256'],'model_catalog_sha256':CATALOG_SHA256}))
if __name__=='__main__':main()
