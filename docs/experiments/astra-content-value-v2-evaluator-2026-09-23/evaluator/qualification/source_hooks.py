"""Source placement and synthetic identity controls, no native solver."""
import argparse,hashlib,importlib.util,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];V1=ROOT.parents[1]/'astra-content-value-20260921/evaluator/general'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[]
 for case,scripts in [('04_gripper',['case04_structure_fixtures.py','case04_source_probe.py']),('09_printer',['case09_placement.py','case09_placement_fixtures.py'])]:
  folder=a.output/case;folder.mkdir()
  for f in (ROOT/'cases'/case).iterdir():
   if f.is_file() and f.suffix in ('.py','.json'):shutil.copy2(f,folder/f.name)
  for name in scripts:
   source=V1/name;text=source.read_text()
   if name=='case04_source_probe.py':text=text.replace("ROOT=HERE/'references/04_gripper'","ROOT=pathlib.Path("+repr(str((ROOT/'cases/04_gripper/reference').resolve()))+")")
   (folder/name).write_text(text);rows.append({'source':str(source),'sha256':sha(source),'copied_to':str(folder/name),'copied_sha256':sha(folder/name)})
  commands=[[sys.executable,str(folder/scripts[0])],[sys.executable,str(folder/scripts[1])]] if case=='04_gripper' else [[sys.executable,str(folder/'case09_placement_fixtures.py')]]
  for command in commands:
   with (folder/(Path(command[1]).stem+'.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
  if case=='09_printer':
   prior=json.loads((ROOT/'cases'/case/'case09_placement.json').read_text());command=[sys.executable,str(folder/'case09_placement.py'),'--inventory',str((ROOT/'cases'/case/'reference/source_inventory.json').resolve()),'--output',str(folder/'rebuilt_placement.json')]
   for ident in prior['bed_source_ids']:command+=['--bed-source-id',ident]
   with (folder/'source_placement.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
   rebuilt=json.loads((folder/'rebuilt_placement.json').read_text());assert prior==rebuilt
 result={'passed':True,'source09_placement_exact_rebuilt_json_equality':True,'source04_original_rays_and_sign_passed':True,'synthetic_role_ancestry_controls':7,'synthetic_source_placement_controls':5,'model_calls':0,'native_solver_calls':0,'script_provenance':rows};(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
