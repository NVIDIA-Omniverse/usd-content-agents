"""Parse actual native subcommands with the shared provider file; no handlers run."""
import argparse
import hashlib
import json
from pathlib import Path
from content_workflow_cli.cli import build_parser, _load_codex_config

def check(provider):
    provider=Path(provider).resolve();value=json.loads(provider.read_text())
    common=['--runner','codex','--model','gpt-6-astra','--model-reasoning-effort','ultra',
            '--codex-config-file',str(provider)]
    rows=[]
    for stage,args in [('physics.apply',['physics','apply','--usd','/work/synthetic.usda','--output-dir','/work/synthetic-physics']),
                       ('validate.run',['validate','run','--usd','/work/synthetic.usda','--task','Synthetic parser-only qualification','--output-dir','/work/synthetic-validation'])]:
        parsed=build_parser().parse_args(args+common)
        assert _load_codex_config(parsed)==value
        assert parsed.model=='gpt-6-astra' and parsed.model_reasoning_effort=='ultra'
        assert parsed.codex_base_url is None
        assert not hasattr(parsed,'codex_responses_url') and not hasattr(parsed,'codex_api_key_env')
        rows.append({'stage':stage,'explicit_provider_file_parsed':True,
          'model':'gpt-6-astra','ui_effort':'ultra','legacy_base_url':None,
          'responses_url_and_key_env_flags_supported':False})
    return {'schema_version':'native-provider-parser.v2','status':'PASS','stages':rows,
      'provider_sha256':hashlib.sha256(provider.read_bytes()).hexdigest(),
      'model_calls':0,'handlers_executed':False,'scope':'Real CLI parsers and config loader only; live child remains separate.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--provider',required=True);a=p.parse_args()
    print(json.dumps(check(a.provider),indent=2))
