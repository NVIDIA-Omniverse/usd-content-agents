"""Trusted USD parser preflight, then the already frozen task evaluator."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def parse(path, output):
    result={'status':'INCONCLUSIVE','scope':'Independent submitted USD parser preflight'}
    try:
        from pxr import Tf, Usd
        if not Path(path).is_file() or Path(path).stat().st_size==0:
            result.update(status='FAIL',reason='Missing or empty declared USD')
        else:
            try:
                stage=Usd.Stage.Open(path)
                result.update(status='PASS' if stage is not None else 'FAIL',reason='USD parser result')
            except Tf.ErrorException as exc:
                result.update(status='FAIL',reason='Submitted USD cannot be parsed: '+str(exc))
    except Exception as exc:
        result['reason']='Evaluator parser unavailable: '+type(exc).__name__+': '+str(exc)
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result['status']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parse-only',action='store_true');p.add_argument('--usd',required=True)
    p.add_argument('--output',required=True);p.add_argument('command',nargs=argparse.REMAINDER);args=p.parse_args()
    if args.parse_only:sys.exit(0 if parse(args.usd,args.output)=='PASS' else 1)
    cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--parse-only','--usd',args.usd,'--output',args.output]
    try:completed=subprocess.run(cmd,timeout=900)
    except subprocess.TimeoutExpired:
        Path(args.output).write_text(json.dumps({'status':'INCONCLUSIVE','reason':'USD parser exceeded frozen 900-second limit'})+'\n')
        sys.exit(2)
    if completed.returncode or not Path(args.output).is_file():sys.exit(2)
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:raise ValueError('Missing trusted frozen evaluator command')
    sys.exit(subprocess.call(command))
