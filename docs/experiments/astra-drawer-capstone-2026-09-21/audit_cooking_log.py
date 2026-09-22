"""Retain native warnings and reject explicit configuration/parser/cooking errors."""
import argparse
import hashlib
import json
import re
from pathlib import Path

def classify(line):
    lower=line.lower()
    if re.search(r'invalid.*(?:must be|parameter|configuration)|\[error\]|\[fatal\]|failed to (?:parse|load)',lower):
        return 'critical_configuration_or_runtime_error'
    if 'gpu-compatible' in lower and ('fall back to cpu' in lower or 'could not be built' in lower):
        return 'cpu_fallback_gpu_collision_unqualified'
    if 'gpu broadphase requires a cuda context' in lower:
        return 'cpu_broadphase_fallback'
    if '[warning]' in lower or '[warn]' in lower or 'warning:' in lower:
        return 'warning_requires_scoped_review'
    return None

def audit(asset,logs):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    records=[]
    for path in logs:
        records.append({'path':str(path),'sha256':sha(path),'messages':[
            {'line':i,'classification':classify(line),'verbatim':line}
            for i,line in enumerate(path.read_text(errors='replace').splitlines(),1) if classify(line)]})
    messages=[m for r in records for m in r['messages']]
    return {'asset':str(asset),'asset_sha256':sha(asset),'logs':records,
            'critical_error_absent':not any(m['classification']=='critical_configuration_or_runtime_error' for m in messages),
            'unclassified_warnings':[m for m in messages if m['classification']=='warning_requires_scoped_review'],
            'scope':'Read-only log gate. A clean configuration log does not establish cooking fidelity, task acceptance, GPU collision, particles or deformables.'}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--asset',type=Path,required=True);ap.add_argument('--log',type=Path,action='append',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert not a.output.exists()
    result=audit(a.asset,a.log);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'critical_error_absent':result['critical_error_absent'],'warnings_for_review':len(result['unclassified_warnings'])}))
    raise SystemExit(0 if result['critical_error_absent'] else 1)
