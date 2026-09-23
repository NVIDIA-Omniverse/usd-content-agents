"""Trusted source unavailability is inconclusive, never an author's failure."""
import importlib.util,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'qualification/reference_disposition_v2';out.mkdir(exist_ok=False)
folder=out/'fixture';folder.mkdir();impl=out/'implementation';impl.mkdir()
for p in (ROOT/'cases/03_hinge').iterdir():
 if p.is_file() and p.suffix in ['.py','.json']:shutil.copy2(p,impl/p.name)
shutil.copy2(ROOT/'qualification/fixture_builders/03_hinge.py',impl/'fixture_builder.py');sys.path.insert(0,str(impl));spec=importlib.util.spec_from_file_location('builder',impl/'fixture_builder.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder);builder.create(folder)
from structural import inspect
from common import dump
bindings=json.loads((folder/'bindings.json').read_text());inv=json.loads((folder/'reference/source_inventory.json').read_text());data=json.loads((impl/'cases.json').read_text());contract=dict(data['cases']['03_hinge'],common=data['common'],case_id='03_hinge');missing=folder/'source'/inv['source_files'][0]['path'];missing.unlink();measurement=out/'measurement';measurement.mkdir()
checks,_=inspect(folder/'positive.usda',bindings,contract,inv,folder/'source',folder/'reference',measurement);failed=[c for c in checks if not c['passed']];assert failed and all(c['failure_class']=='evaluator_insufficient' for c in failed);assert any(c['name'].startswith('source_bytes:') for c in failed)
dump(out/'result.json',{'qualified':True,'unavailable_trusted_source_checks':failed,'author_output_used':False,'model_calls':0,'native_calls':0});print('PASS')
