"""Remove installer caches and refresh RECORD only for known relocated files.

Original installation timestamps/RECORDs remain on the supervisor and in prior
inventories. No native binary or implementation source is edited.
"""
import argparse,base64,csv,hashlib,io,json,os
from pathlib import Path
from export_namespace_tools import sha,manifest,staging_mapping,atomic_bytes

def normalize(root,tools,parent):
 changed={c['path'] for c in parent.get('generated_relocations',[])}|{c['path'] for c in parent.get('locator_repairs',[])}
 removed=set();actions=[]
 def original_for(path):
  rel=path.relative_to(tools);prefix=rel.parts[0];base=root/'repo/.venv' if prefix=='main-venv' else root/prefix
  return base/Path(*rel.parts[1:])
 def record_change(path,before,kind):
  original=original_for(path);original_digest=sha(original)
  actions.append({'path':str(path.relative_to(tools)),'kind':kind,'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':sha(path) if path.exists() else None,'original_installed_sha256':original_digest,'original_unchanged':True})
 for venv in ['main-venv','ovphysx-venv','ovrtx-venv']:
  for cache in sorted((tools/venv/'lib/python3.12/site-packages').glob('*.dist-info/uv_cache.json')):
   before=cache.read_bytes();payload=json.loads(before)
   if not isinstance(payload,dict) or 'timestamp' not in payload:raise ValueError('Unrecognized installer cache')
   original=original_for(cache);digest=sha(original);cache.unlink()
   if sha(original)!=digest:raise ValueError('Original cache changed')
   removed.add(str(cache.relative_to(tools)));record_change(cache,before,'remove_installer_timestamp_cache')
 mappings=staging_mapping(root)
 for name in ['activate','activate.csh','activate.fish']:
  path=tools/'ovrtx-venv/bin'/name;before=path.read_bytes();after=before
  for old,_target in mappings:after=after.replace(Path(old).name.encode(),b'ovrtx-venv')
  if after!=before:
   original=original_for(path);digest=sha(original);atomic_bytes(path,after)
   if sha(original)!=digest:raise ValueError('Original activation script changed')
   changed.add(str(path.relative_to(tools)));record_change(path,before,'normalize_activation_prompt_label')
 for venv in ['main-venv','ovphysx-venv','ovrtx-venv']:
  for record in sorted((tools/venv/'lib/python3.12/site-packages').glob('*.dist-info/RECORD')):
   before=record.read_bytes();rows=list(csv.reader(io.StringIO(before.decode())));updated=[];touched=False
   for row in rows:
    if len(row)!=3:raise ValueError('Malformed RECORD')
    path=Path(os.path.normpath(str(record.parent.parent/row[0])))
    if not path.is_relative_to(tools):raise ValueError('RECORD path outside export')
    rel=str(path.relative_to(tools))
    if rel in removed:touched=True;continue
    if rel in changed and path.is_file() and not path.is_symlink():
     digest='sha256='+base64.urlsafe_b64encode(bytes.fromhex(sha(path))).decode().rstrip('=');new=[row[0],digest,str(path.stat().st_size)];touched=touched or row!=new;row=new
    updated.append(row)
   if touched:
    text=io.StringIO(newline='');csv.writer(text,lineterminator='\n').writerows(updated);original=original_for(record);digest=sha(original);atomic_bytes(record,text.getvalue().encode())
    if sha(original)!=digest:raise ValueError('Original RECORD changed')
    record_change(record,before,'refresh_only_relocated_entries_and_removed_cache')
 return actions

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--tools',type=Path,required=True);p.add_argument('--parent-receipt',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();root=a.root.resolve();tools=a.tools.resolve()
 if a.receipt.exists():raise ValueError('Create-only metadata receipt')
 parent=json.loads(a.parent_receipt.read_text());before={f['path']:f for f in manifest(tools)};expected={f['path']:f for f in parent['files']}
 if not parent['passed'] or any(before.get(p)!=v for p,v in expected.items() if not p.startswith('repo/.git/')) or any(p not in expected for p in before if not p.startswith('repo/.git/')):raise ValueError('Stale parent inventory')
 actions=normalize(root,tools,parent);files=manifest(tools);result=dict(parent);result.update(parent_receipt_sha256=sha(a.parent_receipt),metadata_normalizer_sha256=sha(Path(__file__)),metadata_normalization=actions,files=files,manifest_sha256=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),qualification='EXPORT_METADATA_NORMALIZED_RUNTIME_AND_SOURCE_UNCHANGED')
 a.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'passed':True,'metadata_actions':len(actions),'manifest_sha256':result['manifest_sha256']}))
if __name__=='__main__':main()
