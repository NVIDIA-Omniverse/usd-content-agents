"""Relocate exact submitted bytes and check USD asset resolution, without editing them."""
from pathlib import Path
import argparse,datetime,hashlib,json,shutil
from pxr import Sdf,Usd,UsdUtils
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
assert not a.destination.exists(),'Use a fresh destination'
source=a.source.resolve();dest=a.destination.resolve();dest.mkdir(parents=True)
copied=[]
for f in [source/'final.usd',source/'bindings.json',*sorted((source/'assets').rglob('*'))]:
    if not f.is_file():continue
    assert not f.is_symlink()
    rel=f.relative_to(source);out=dest/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,out)
    assert sha(f)==sha(out);copied.append({'path':str(rel),'sha256':sha(out),'bytes':out.stat().st_size})
stage=Usd.Stage.Open(str(dest/'final.usd'));assert stage
assets=[]
for prim in stage.Traverse():
    for attr in prim.GetAttributes():
        value=attr.Get()
        if isinstance(value,Sdf.AssetPath) and value.path:
            resolved=Path(value.resolvedPath).resolve() if value.resolvedPath else None
            assets.append({'attribute':str(attr.GetPath()),'authored_path':value.path,'relative':not Path(value.path).is_absolute(),'resolved_inside_relocation':bool(resolved and resolved.is_file() and resolved.is_relative_to(dest)),'sha256':sha(resolved) if resolved and resolved.is_file() else None})
layers,external,unresolved=UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(dest/'final.usd')))
bindings=json.loads((dest/'bindings.json').read_text());bound=dest/bindings['final_usd']
ok=bool(assets) and all(x['relative'] and x['resolved_inside_relocation'] for x in assets) and not unresolved and bound.resolve()==(dest/'final.usd') and all(Path(x).resolve().is_relative_to(dest) for x in external) and all(sha(source/x['path'])==x['sha256'] for x in copied)
result={'schema_version':'drawer-submission-portability.v1','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':ok,'source':str(source),'relocation':str(dest),'copied_bytes_unchanged':copied,'asset_attributes':assets,'external_dependencies':[str(Path(x).resolve().relative_to(dest)) for x in external],'unresolved_dependencies':list(unresolved),'binding_scene_resolves':bound.resolve()==(dest/'final.usd'),'wrapper_needed':not ok,'publication_layout':'Keep final.usd and bindings.json beside the complete assets/textures directory. The original scored USD and binding bytes need no rewrite.','scope':'USD dependency resolution of exact relocated bytes, not a new physical evaluation or render.'}
(dest/'portability_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));raise SystemExit(0 if ok else 1)
