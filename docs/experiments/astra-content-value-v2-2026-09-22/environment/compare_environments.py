"""Exact semantic parity gate; outcomes and readiness cannot substitute for parity."""
import argparse,json
from pathlib import Path
from record_runtime import semantic

def compare(a,b):
 differences=[k for k in semantic(a) if semantic(a)[k]!=semantic(b)[k]]
 return {'schema_version':'matched-environment-comparison.v2','passed':a.get('passed') is True and b.get('passed') is True and not differences,'different_fields':differences,'both_import_preflights_pass':a.get('passed') is True and b.get('passed') is True,'end_to_end_workflow_qualified':False}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('first',type=Path);p.add_argument('second',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=compare(json.loads(a.first.read_text()),json.loads(a.second.read_text()));a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));raise SystemExit(0 if r['passed'] else 2)
