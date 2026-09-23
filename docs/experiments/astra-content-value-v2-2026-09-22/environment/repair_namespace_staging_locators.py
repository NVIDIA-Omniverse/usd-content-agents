"""Bounded repair for generated console scripts missed by older exporter.
No binary/module/source edits; retain exact parent manifest and before/after SHA.
"""
import argparse,hashlib,json
from pathlib import Path
from export_namespace_tools import sha,manifest,staging_mapping,atomic_bytes
ALLOWED={'activate','activate.csh','activate.fish','pip','pip3','pip3.12','f2py','numpy-config'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--tools',type=Path,required=True);p.add_argument('--parent-receipt',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();tools=a.tools.resolve();root=a.root.resolve()
 if a.receipt.exists():raise ValueError('Repair receipt is create-only')
 parent=json.loads(a.parent_receipt.read_text());before={f['path']:f for f in manifest(tools)};expected={f['path']:f for f in parent['files']}
 if not parent['passed'] or any(before.get(p)!=v for p,v in expected.items() if not p.startswith('repo/.git/')):raise ValueError('Parent inventory stale')
 if any(p not in expected for p in before if not p.startswith('repo/.git/')):raise ValueError('Unexpected installed file')
 mapping=staging_mapping(root);changes=[]
 for path in sorted((tools/'ovrtx-venv/bin').iterdir()):
  if path.is_symlink() or not path.is_file():continue
  data=path.read_bytes();new=data
  for old,target in mapping:new=new.replace(old.encode(),target.encode())
  if new!=data:
   if path.name not in ALLOWED:raise ValueError('Unexpected staging locator file '+path.name)
   original=root/'ovrtx-venv/bin'/path.name;original_sha=sha(original);atomic_bytes(path,new)
   if sha(original)!=original_sha:raise ValueError('Original hardlink was modified')
   changes.append({'path':str(path.relative_to(tools)),'before_sha256':hashlib.sha256(data).hexdigest(),'after_sha256':sha(path),'original_installed_sha256':original_sha,'original_unchanged':True})
 files=manifest(tools);result=dict(parent);result.update(parent_receipt_sha256=sha(a.parent_receipt),locator_repair_script_sha256=sha(Path(__file__)),locator_repairs=changes,files=files,manifest_sha256=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),qualification='REQUALIFY_NAMESPACE_AFTER_GENERATED_LOCATOR_REPAIR')
 a.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'passed':True,'repairs':len(changes),'manifest_sha256':result['manifest_sha256']}))
if __name__=='__main__':main()
