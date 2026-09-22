"""One post-freeze controller diagnostic; never an acceptance replacement."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.frozen))
    spec=importlib.util.spec_from_file_location('unchanged_frozen_solver',a.frozen/'solver.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_np=module.np
    calls=[]
    class Generator:
        def __init__(self,seed):self.generator=original_np.random.default_rng(seed)
        def uniform(self,low=0.0,high=1.0,size=None):
            if size is None and (low,high) in ((.98,1.02),(-.005,.005)):
                value=1.0 if low==.98 else 0.0
                calls.append({'low':low,'high':high,'replacement':value})
                return value
            return self.generator.uniform(low,high,size)
        def __getattr__(self,name):return getattr(self.generator,name)
    class Random:
        def default_rng(self,seed=None):return Generator(seed)
        def __getattr__(self,name):return getattr(original_np.random,name)
    class Numpy:
        random=Random()
        def __getattr__(self,name):return getattr(original_np,name)
    # Only this imported solver's namespace changes. Native library modules
    # keep their original NumPy module and all frozen files remain unchanged.
    module.np=Numpy()
    before={'solver_sha256':sha(a.frozen/'solver.py'),'config_sha256':sha(a.config)}
    start=time.time()
    module.run(json.loads(a.config.read_text()),11,a.output/'seed11_no_jitter.json',device='cpu')
    after={'solver_sha256':sha(a.frozen/'solver.py'),'config_sha256':sha(a.config)}
    receipt={'schema_version':1,'scope':'Unscored single-seed controller diagnostic. It sets effort multiplier to1 and startup random effort to0; it cannot confer complete asset acceptance.','seed':11,'before':before,'after':after,'frozen_solver_and_config_unchanged':before==after,'replacement_counts':{'effort_multiplier':sum(x['replacement']==1 for x in calls),'startup_perturbations':sum(x['replacement']==0 for x in calls)},'diagnostic_sha256':sha(Path(__file__)),'native_result_sha256':sha(a.output/'seed11_no_jitter.json'),'elapsed_seconds':time.time()-start}
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
