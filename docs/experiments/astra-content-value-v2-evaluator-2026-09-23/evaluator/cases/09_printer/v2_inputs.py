"""Reject concrete invalid author inputs before evaluator-owned work starts."""
import json
from pathlib import Path


def _nonfinite(value):
    raise ValueError('Nonfinite JSON constant: '+value)


def author_input_preflight(args, here):
    from common import check, dump, verdict
    from pxr import Tf, Usd
    from v2_process import inspect_with_deadline
    import jsonschema
    checks=[]
    for label,path in [('submitted_usd',args.usd),('submitted_bindings',args.bindings)]:
        try:
            if not Path(path).is_file():
                check(checks,label+'_file',False,'Required author artifact is missing or not a regular file')
            elif Path(path).stat().st_size==0:
                check(checks,label+'_file',False,'Required author artifact is empty')
        except OSError as exc:
            check(checks,label+'_accessible',False,type(exc).__name__+': '+str(exc),True)
    if not checks:
        try:
            bindings=json.loads(Path(args.bindings).read_text(),parse_constant=_nonfinite)
        except (ValueError,UnicodeError) as exc:
            check(checks,'bindings_json',False,type(exc).__name__+': '+str(exc))
        except OSError as exc:
            check(checks,'bindings_accessible',False,type(exc).__name__+': '+str(exc),True)
        else:
            try:
                schema=json.loads((Path(here)/'bindings.schema.json').read_text())
                jsonschema.validate(bindings,schema)
            except jsonschema.ValidationError as exc:
                check(checks,'bindings_schema',False,str(exc))
            except Exception as exc:
                check(checks,'evaluator_schema_available',False,type(exc).__name__+': '+str(exc),True)
        try:
            stage=inspect_with_deadline(Usd.Stage.Open,str(args.usd),timeout_s=900)
            check(checks,'usd_stage_load',stage is not None)
        except Tf.ErrorException as exc:
            check(checks,'usd_stage_load',False,str(exc))
        except Exception as exc:
            check(checks,'usd_loader_available',False,type(exc).__name__+': '+str(exc),True)
    if any(not row['passed'] for row in checks):
        args.output.mkdir(parents=True,exist_ok=False)
        report=verdict(checks)
        report.update(case_id=getattr(args,'case',Path(here).name),
                      input_disposition='Author file/JSON/schema/USD failures are concrete rejection; evaluator access or configuration failures are inconclusive.')
        dump(args.output/'acceptance.json',report)
        return False
    return True
