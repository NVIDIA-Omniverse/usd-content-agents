"""Compare actual installed export bytes; Git administrative timestamps excluded.

Retained source bytes and the parentless commit are both checked by the exporter.
This comparison never substitutes for actual per-lane namespace qualification.
"""
import argparse,hashlib,json
from pathlib import Path
def semantic(receipt):
 return {'upstream_commit':receipt['upstream_commit'],'source_git_commit':receipt['source_git_commit'],'files':{f['path']:{k:v for k,v in f.items() if k!='path'} for f in receipt['files'] if not f['path'].startswith('repo/.git/')}}
def compare(a,b):
 aa=semantic(a);bb=semantic(b);paths=set(aa['files'])|set(bb['files']);different=[p for p in sorted(paths) if aa['files'].get(p)!=bb['files'].get(p)]
 same_identity=aa['upstream_commit']==bb['upstream_commit'] and aa['source_git_commit']==bb['source_git_commit']
 return {'schema_version':'namespace-tools-parity.v2','passed':bool(a['passed'] and b['passed'] and same_identity and not different),'source_identity_equal':same_identity,'different_file_paths':different,'compared_file_count':len(paths),'excluded':'repo/.git administrative bytes only; clean tracked source and identical parentless commit remain mandatory','end_to_end_workflow_qualified':False,'a_semantic_sha256':hashlib.sha256(json.dumps(aa,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'b_semantic_sha256':hashlib.sha256(json.dumps(bb,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('a',type=Path);p.add_argument('b',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=compare(json.loads(a.a.read_text()),json.loads(a.b.read_text()));a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));raise SystemExit(0 if r['passed'] else 2)
