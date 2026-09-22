#!/usr/bin/env python3
"""Reindex freshly rebuilt reference bytes using unchanged frozen SHA256 values.

This is an explicit compatibility step, not a modification of the frozen importer,
geometry or expected inventory. Raw rebuilt outputs remain available for audit.
"""
import argparse, collections, datetime, hashlib, json, shutil
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--rebuilt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compare-original-arrays',action='store_true',help='Optional qualification check requiring retained original NPZ files; not needed for public regeneration.')
    a=p.parse_args();original=a.inventory.resolve();inv=json.loads(original.read_text());inventory_sha=sha(original)
    if a.compare_original_arrays:import numpy as np
    rebuilt=a.rebuilt.resolve();generated=json.loads((rebuilt/'source_inventory.json').read_text())
    expected=collections.Counter(x['geometry_sha256'] for x in inv['parts']);found=collections.defaultdict(list)
    for part in generated['parts']:found[sha(rebuilt/part['geometry_file'])].append(part)
    actual=collections.Counter({k:len(v) for k,v in found.items()})
    if expected!=actual:raise SystemExit('Cannot align: frozen and rebuilt NPZ hash multisets differ. Expected hashes are unchanged.')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);mappings=[]
    for part in inv['parts']:
        digest=part['geometry_sha256'];candidates=found[digest];match=candidates.pop(0)
        src=rebuilt/match['geometry_file'];dest=out/part['geometry_file'];assert dest.parent==out
        shutil.copyfile(src,dest);assert sha(dest)==digest
        arrays_equal=None
        if a.compare_original_arrays:
            with np.load(original.parent/part['geometry_file'],allow_pickle=False) as x,np.load(dest,allow_pickle=False) as y:
                arrays_equal=x.files==y.files and all(x[k].dtype==y[k].dtype and np.array_equal(x[k],y[k],equal_nan=True) for k in x.files)
            assert arrays_equal
        mappings.append({'frozen_source_id':part['source_id'],'rebuilt_source_id':match['source_id'],'rebuilt_file':match['geometry_file'],'frozen_file':part['geometry_file'],'sha256':digest,'all_arrays_exactly_equal':arrays_equal})
    shutil.copyfile(original,out/'source_inventory.json');assert sha(out/'source_inventory.json')==inventory_sha
    result={'case_id':inv['case_id'],'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_inventory_sha256':inventory_sha,'aligned_inventory_sha256':sha(out/'source_inventory.json'),'part_count':len(mappings),'same_npz_hash_multiset':True,'every_aligned_npz_matches_frozen_hash':True,'array_comparison_performed':a.compare_original_arrays,'every_array_exactly_equal':True if a.compare_original_arrays else None,'mappings':mappings,'operation':'Only copy exact regenerated NPZ bytes to filenames selected by original expected SHA256; copy original complete inventory unchanged. Do not serialize arrays, modify importer code, or rewrite expected hashes.','scope':'Reference-reader byte compatibility established in this recorded runtime. Default operation needs the original inventory only, not original NPZ files. No authored asset or physical evaluation is read or rerun.'}
    (out/'alignment_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='mappings'},indent=2))
if __name__=='__main__':main()
