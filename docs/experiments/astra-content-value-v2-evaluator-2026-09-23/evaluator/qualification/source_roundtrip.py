"""Full original-reference performance check; no authored asset is consumed."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'common'))
from v2_geometry import surface_compare


def main():
    p = argparse.ArgumentParser();p.add_argument('--case', default='09_printer');p.add_argument('--mode', choices=['float32','subdivide'], default='float32');p.add_argument('--limit',type=int);p.add_argument('--output', type=Path, required=True);a=p.parse_args()
    reference = ROOT / 'cases' / a.case / 'reference'
    invpath = reference / 'source_inventory.json';inv=json.loads(invpath.read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    tick=time.monotonic();passed=True;count=0;methods={}
    code_sha=hashlib.sha256((ROOT/'common/v2_geometry.py').read_bytes()).hexdigest()
    with (a.output/'parts.jsonl').open('w') as trace:
        for part in inv['parts']:
            if a.limit and count>=a.limit:break
            path=reference/part['geometry_file']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==part['geometry_sha256']
            with np.load(path) as arrays:
                v,f=arrays['vertices'],arrays['faces']
                # USD point3f round trip is a source-derived numerical fixture,
                # never a simulated/solved mechanism or an author input.
                other=v.astype(np.float32).astype(np.float64);other_f=f
                if a.mode=='subdivide':
                    import trimesh
                    other,other_f=trimesh.remesh.subdivide(other,f)
                tol=max(5e-5, .001*float(np.linalg.norm(np.ptp(v,axis=0))))
                ok,detail=surface_compare(v,f,other,other_f,tol)
            passed &= ok;count+=1;methods[detail['method']]=methods.get(detail['method'],0)+1
            trace.write(json.dumps({'source_id':part['source_id'],'passed':ok,'measurement':detail})+'\n')
    result={'case_id':a.case,'mode':a.mode,'scope':'Every independently frozen original mesh, float32 roundtrip with optional triangle subdivision; not stage ingestion or full task performance.',
            'passed':bool(passed),'part_count':count,'full_inventory':count==len(inv['parts']),'elapsed_s':time.monotonic()-tick,'methods':methods,'comparison_code_sha256':code_sha,
            'inventory_sha256':hashlib.sha256(invpath.read_bytes()).hexdigest(), 'author_output_used':False}
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    raise SystemExit(0 if passed else 1)


if __name__=='__main__':main()
