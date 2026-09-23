"""Actual CLI negative controls, never native solver or author assets."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
CASES=sorted(p.name for p in (ROOT/'cases').iterdir() if p.is_dir())

@pytest.mark.parametrize('case',CASES)
@pytest.mark.parametrize('variant,expected',[('missing_usd','submitted_usd_file'),('missing_bindings','submitted_bindings_file'),('bad_json','bindings_json'),('bad_schema','bindings_schema'),('bad_usd','usd_stage_load'),('nonfinite_json','bindings_json')])
def test_bad_author_input_is_concrete_failure(tmp_path,case,variant,expected):
    usd=tmp_path/'final.usda';usd.write_text('#usda 1.0\n')
    bindings=tmp_path/'bindings.json';bindings.write_text('{}')
    if variant=='missing_usd':usd.unlink()
    elif variant=='missing_bindings':bindings.unlink()
    elif variant=='bad_json':bindings.write_text('{')
    elif variant=='bad_usd':usd.write_text('malformed USDA')
    elif variant=='nonfinite_json':bindings.write_text('{"value":NaN}')
    output=tmp_path/'report'
    command=[sys.executable,str(ROOT/'cases'/case/'evaluate.py'),'--usd',str(usd),'--bindings',str(bindings),'--inventory',str(tmp_path/'unavailable_evaluator_inventory.json'),'--source-root',str(tmp_path/'unavailable_source'),'--output',str(output)]
    if case!='02_conveyor':command+=['--case',case,'--structural-only']
    run=subprocess.run(command,capture_output=True,text=True,timeout=30)
    assert run.returncode==0,run.stderr
    report=json.loads((output/'acceptance.json').read_text())
    assert not report['accepted'] and report['status']=='not_accepted'
    assert expected in report['concrete_failures']
    assert not report['inconclusive_checks']

@pytest.mark.parametrize('case',CASES)
def test_public_inventory_path_and_no_stale_hash(case):
    text=(ROOT/'tasks'/(case+'.json')).read_text();task=json.loads(text)
    assert '/input/'+case+'_source_inventory.json' in text
    assert '/protocol/tasks/' not in text
    assert 'evaluator_freeze_sha256' not in task
    assert type(task['frozen']) is bool
    if task['frozen']:
        assert len(task['private_evaluator_snapshot_sha256']) == 64

@pytest.mark.parametrize('mode',['valid','timeout'])
def test_input_stage_load_deadline_disposition(tmp_path,monkeypatch,mode):
    import importlib.util
    from types import SimpleNamespace
    case=ROOT/'cases/03_hinge';monkeypatch.syspath_prepend(str(case))
    import v2_process
    spec=importlib.util.spec_from_file_location('input_gate',case/'v2_inputs.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    bindings=json.loads((ROOT/'qualification/local_preparation/03_hinge/generated/bindings.json').read_text())
    b=tmp_path/'bindings.json';b.write_text(json.dumps(bindings));usd=tmp_path/'good.usda';usd.write_text('#usda 1.0\n');out=tmp_path/'report'
    if mode=='timeout':
        def unavailable(*args,**kwargs):
            assert kwargs['timeout_s']==900
            raise v2_process.InspectionDeadline('Synthetic input stage deadline')
        monkeypatch.setattr(v2_process,'inspect_with_deadline',unavailable)
    okay=module.author_input_preflight(SimpleNamespace(usd=usd,bindings=b,output=out,case='03_hinge'),case)
    assert okay==(mode=='valid')
    if mode=='timeout':
        report=json.loads((out/'acceptance.json').read_text());assert report['status']=='inconclusive' and not report['concrete_failures']

@pytest.mark.parametrize('case',CASES)
def test_public_inventory_hash_is_public_file(case):
    import hashlib
    task=json.loads((ROOT/'tasks'/(case+'.json')).read_text())
    assert task['source_inventory_file']=='/input/'+case+'_source_inventory.json'
    assert task['source_inventory_sha256']==hashlib.sha256((ROOT/'tasks'/(case+'_source_inventory.json')).read_bytes()).hexdigest()
