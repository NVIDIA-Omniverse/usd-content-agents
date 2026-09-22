#!/usr/bin/env python3
"""Rebuild original source references with immutable case code; no author reads."""
import argparse, datetime, hashlib, importlib.metadata, json, os
from pathlib import Path
import platform, shutil, subprocess, sys, time, zipfile, zlib

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--case',choices=['03_hinge','05_vise'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout',type=int,default=180)
    a=p.parse_args();root=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.time();started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    # One CPU maximum, inherited by the frozen importer including native workers.
    affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{affinity[-1]})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    frozen=root/'evaluator/general/frozen'/a.case;reference=root/'evaluator/general/references'/a.case
    manifest=json.loads((frozen/'frozen_manifest.json').read_text());original=reference/'source_inventory.json';inv=json.loads(original.read_text())
    inv_sha=sha(original);assert inv_sha==manifest['reference_inventory_sha256']
    code_before={n:sha(frozen/n) for n in manifest['code_sha256']};assert code_before==manifest['code_sha256']
    source=root/'assets'/a.case
    source_before={x['path']:sha(source/x['path']) for x in inv['source_files']}
    assert all(source_before[x['path']]==x['sha256'] for x in inv['source_files'])
    versions={k:importlib.metadata.version(k) for k in ['numpy','trimesh','cadquery-ocp']}
    assert all(versions[k]==v for k,v in inv['importer_versions'].items()),(versions,inv['importer_versions'])
    rebuilt=out/'rebuilt';command=[sys.executable,str(frozen/'geometry.py'),'--asset-root',str(source),'--source',inv['primary_input'],'--unit',str(inv['length_scale_to_m']),'--output',str(rebuilt),'--case',a.case]
    for ident in inv.get('excluded_ids',[]):command.extend(['--exclude-id',ident])
    dump(out/'launch.json',{'case_id':a.case,'started_utc':started_utc,'argv':command,'cwd':str(out),'environment_overrides':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']},'cpu_affinity':[affinity[-1]],'nice':os.nice(0),'python':sys.version,'platform':platform.platform(),'zlib':zlib.ZLIB_RUNTIME_VERSION,'versions':versions,'original_inventory_sha256':inv_sha,'frozen_manifest_sha256':sha(frozen/'frozen_manifest.json'),'code_sha256':code_before,'source_sha256':source_before,'scope':'Original source and evaluator preparation only; no author outputs read.'})
    process_start=time.time()
    with (out/'freezer.stdout.log').open('w') as stdout,(out/'freezer.stderr.log').open('w') as stderr:
        try:proc=subprocess.run(command,cwd=out,env=env,stdout=stdout,stderr=stderr,timeout=a.timeout);returncode=proc.returncode;error=None
        except subprocess.TimeoutExpired: returncode=None;error='frozen_source_import_timeout'
    duration=time.time()-process_start
    result={'case_id':a.case,'started_utc':started_utc,'import_duration_s':duration,'returncode':returncode,'error':error,'expected_parts':len(inv['parts']),'versions':versions,'original_inventory_sha256':inv_sha,'parts':[]}
    if returncode==0:
        import numpy as np
        generated=json.loads((rebuilt/'source_inventory.json').read_text())
        result['regenerated_parts']=len(generated['parts'])
        result['source_id_order_equal']=[x['source_id'] for x in generated['parts']]==[x['source_id'] for x in inv['parts']]
        for part in inv['parts']:
            old=reference/part['geometry_file'];new=rebuilt/part['geometry_file']
            record={'source_id':part['source_id'],'geometry_file':part['geometry_file'],'expected_sha256':part['geometry_sha256'],'original_sha256':sha(old),'rebuilt_sha256':sha(new),'arrays':{}}
            record['original_matches_inventory']=record['original_sha256']==record['expected_sha256']
            record['rebuilt_matches_inventory']=record['rebuilt_sha256']==record['expected_sha256']
            with np.load(old,allow_pickle=False) as x,np.load(new,allow_pickle=False) as y:
                record['array_keys_equal']=x.files==y.files
                for key in x.files:
                    left=x[key];right=y[key];equal=left.shape==right.shape and np.array_equal(left,right,equal_nan=True)
                    entry={'shape':list(left.shape),'dtype':str(left.dtype),'rebuilt_shape':list(right.shape),'rebuilt_dtype':str(right.dtype),'values_equal':bool(equal),'dtype_equal':left.dtype==right.dtype,'original_array_sha256':hashlib.sha256(left.tobytes(order='C')).hexdigest(),'rebuilt_array_sha256':hashlib.sha256(right.tobytes(order='C')).hexdigest()}
                    if not equal and left.shape==right.shape:entry['max_abs_difference']=float(np.max(np.abs(left.astype(float)-right.astype(float))))
                    record['arrays'][key]=entry
            if not record['rebuilt_matches_inventory']:
                def zip_metadata(path):
                    with zipfile.ZipFile(path) as z:return [{'filename':x.filename,'date_time':x.date_time,'compress_type':x.compress_type,'CRC':x.CRC,'file_size':x.file_size,'compress_size':x.compress_size,'create_system':x.create_system,'external_attr':x.external_attr} for x in z.infolist()]
                record['original_zip_metadata']=zip_metadata(old);record['rebuilt_zip_metadata']=zip_metadata(new)
            result['parts'].append(record)
        result['all_npz_byte_identical']=all(x['rebuilt_matches_inventory'] for x in result['parts'])
        result['all_arrays_exactly_equal']=all(x['array_keys_equal'] and all(y['values_equal'] and y['dtype_equal'] for y in x['arrays'].values()) for x in result['parts'])
        # Preserve the regenerated metadata AND a byte-identical copy of the complete frozen inventory.
        shutil.copyfile(original,rebuilt/'published_frozen_inventory.json')
        result['copied_original_inventory_sha256']=sha(rebuilt/'published_frozen_inventory.json')
        result['original_inventory_plus_rebuilt_npz_compatible']=result['source_id_order_equal'] and result['all_npz_byte_identical'] and result['copied_original_inventory_sha256']==inv_sha
    result['original_inventory_unchanged']=sha(original)==inv_sha
    result['frozen_code_unchanged']={n:sha(frozen/n) for n in code_before}==code_before
    result['original_source_closure_unchanged']={n:sha(source/n) for n in source_before}==source_before
    result['elapsed_seconds']=time.time()-started;result['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    result['scope_limit']='Exact regeneration was tested in the recorded existing pinned Linux runtime, on original source bytes. This is not a new-environment installation or redownload test, does not establish cross-version/platform byte stability, and does not rerun author physics. Original inventory hashes were never rewritten.'
    dump(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['parts','versions','scope_limit']},indent=2))

if __name__=='__main__':main()
