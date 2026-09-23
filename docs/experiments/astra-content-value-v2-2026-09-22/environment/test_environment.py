import copy,hashlib,tempfile,unittest
from pathlib import Path
from compare_environments import compare
from emit_environment import environment
from export_namespace_tools import copy_tree,relocate_tree,atomic_bytes,staging_mapping
from make_author_repo import safe_member,excluded
from compare_namespace_exports import compare as compare_exports
import os,tarfile
import json,sys
from unittest.mock import patch
import supplement_namespace_tools as supplement
from normalize_export_metadata import normalize
import csv,io,base64

class Checks(unittest.TestCase):
 def specimen(self):
  return dict(code_commit='pinned',code_tree='tree',python='3.12.11',architecture='x86_64',packages={'usd-exchange':'2.3.1'},pxr_version=[0,25,5],native_resource_files={'binary':'sha'},runtime_lock_files={'lock':'sha'},source_skill_targets={'geometry':'sha'},worker_availability={'geogram':{'available':True}},node_packages={'codex':'0.154.0'},passed=True)
 def test_equal_requires_passed(self):
  a=self.specimen();self.assertTrue(compare(a,a)['passed']);b=copy.deepcopy(a);b['passed']=False;self.assertFalse(compare(a,b)['passed'])
 def test_every_material_runtime_difference_blocks(self):
  a=self.specimen()
  for key in ['code_commit','code_tree','python','architecture','packages','pxr_version','native_resource_files','runtime_lock_files','source_skill_targets','worker_availability','node_packages']:
   with self.subTest(key=key):
    b=copy.deepcopy(a);b[key]='different';r=compare(a,b);self.assertFalse(r['passed']);self.assertIn(key,r['different_fields'])
 def test_parity_cannot_claim_end_to_end(self):
  self.assertFalse(compare(self.specimen(),self.specimen())['end_to_end_workflow_qualified'])
 def test_geogram_digest_must_be_explicit_and_exact(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p=root/'resources/geogram/bin/vorpalite';p.parent.mkdir(parents=True);p.write_bytes(b'synthetic')
   with self.assertRaises(ValueError):environment(root,'0'*64)
   digest=hashlib.sha256(p.read_bytes()).hexdigest();env=environment(root,digest);self.assertEqual(env['GEOMETRY_REPAIR_GEOGRAM_EXECUTABLE_SHA256'],digest)
   self.assertNotIn('CUDA_VISIBLE_DEVICES',env);self.assertNotIn('USD_CLI_SESSION',env);self.assertFalse(any('AUTH' in k or 'KEY' in k for k in env))
 def test_hardlink_relocation_preserves_original_and_code(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);src=root/'src';src.mkdir();(src/'binding.pth').write_text('/host/repo\n');(src/'implementation.py').write_text("LOCATOR='/host/repo'\n");(src/'binding.pth').chmod(0o644)
   target=root/'copy';copy_tree(src,target)
   self.assertEqual(os.stat(src/'binding.pth').st_ino,os.stat(target/'binding.pth').st_ino)
   changes=relocate_tree(target,[('/host/repo','/tools/repo')])
   self.assertEqual((src/'binding.pth').read_text(),'/host/repo\n');self.assertEqual((target/'binding.pth').read_text(),'/tools/repo\n')
   self.assertNotEqual(os.stat(src/'binding.pth').st_ino,os.stat(target/'binding.pth').st_ino)
   self.assertEqual((src/'implementation.py').read_bytes(),(target/'implementation.py').read_bytes());self.assertEqual(len(changes),1)
 def test_staging_directory_permissions_do_not_mutate_original(self):
  with tempfile.TemporaryDirectory() as d:
   src=Path(d)/'src';src.mkdir(mode=0o700);(src/'module.py').write_text('pass\n');dst=Path(d)/'dst';copy_tree(src,dst)
   self.assertEqual(src.stat().st_mode&0o777,0o700);self.assertEqual(dst.stat().st_mode&0o555,0o555)
 def test_absolute_interpreter_symlink_is_relocated(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'python';p.symlink_to('/host/python/bin/python3.12');changes=relocate_tree(Path(d),[('/host/python','/tools/python')]);self.assertEqual(os.readlink(p),'/tools/python/bin/python3.12');self.assertEqual(changes[0]['kind'],'symlink')
 def test_source_archive_cannot_escape_or_import_special_files(self):
  for path,target,allowed in [('good/file',None,True),('../secret',None,False),('/absolute',None,False),('folder/link','../../secret',False),('folder/link','../inside',True),('folder/link','/secret',False)]:
   with self.subTest(path=path,target=target):
    member=tarfile.TarInfo(path)
    if target is not None:member.type=tarfile.SYMTYPE;member.linkname=target
    self.assertEqual(safe_member(member),allowed)
  member=tarfile.TarInfo('device');member.type=tarfile.CHRTYPE;self.assertFalse(safe_member(member))
 def test_source_only_keeps_implementation_and_skills(self):
  for name in ['tests/test_drawer.py','docs/experiments/results.json','agentic/docs/drawer_geometry_capstone.md']:self.assertTrue(excluded(name))
  for name in ['AGENTS.md','agentic/packages/content_agent_workflows/physics/workflow.py','.agents/skills/content-workflow-physics','world_understanding/agentic/validation_scaffold.py']:self.assertFalse(excluded(name))
 def test_python_generated_libdir_and_staging_command_are_portable(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'_sysconfigdata__linux.py').write_text("build_time_vars={'LIBDIR':'/host/python/lib'}\n");(root/'pyvenv.cfg').write_text('home = /host/python/bin\ncommand = /host/python -m venv /host/.random-staging-ABC\nversion = 3.12.11\n')
   relocate_tree(root,[('/host/python','/tools/python')]);self.assertIn('/tools/python/lib',(root/'_sysconfigdata__linux.py').read_text());self.assertNotIn('command =',(root/'pyvenv.cfg').read_text());self.assertNotIn('/host/',(root/'pyvenv.cfg').read_text())
 def test_export_parity_ignores_only_git_administration(self):
  a={'passed':True,'upstream_commit':'pinned','source_git_commit':'derived','files':[{'path':'repo/.git/index','sha256':'old'},{'path':'main-venv/lib/native.so','sha256':'native'},{'path':'repo/workflow.py','sha256':'source'}]};b=copy.deepcopy(a);b['files'][0]['sha256']='fresh';self.assertTrue(compare_exports(a,b)['passed'])
  b['files'][1]['sha256']='forged';self.assertFalse(compare_exports(a,b)['passed']);b=copy.deepcopy(a);b['source_git_commit']='changed';self.assertFalse(compare_exports(a,b)['passed']);b=copy.deepcopy(a);b['passed']=False;self.assertFalse(compare_exports(a,b)['passed'])
 def test_supplement_preserves_code_and_rejects_bad_catalog_or_stale_parent(self):
  for condition in ['valid','catalog_changed','code_changed','failed_parent','existing_conflict','unexpected_extra']:
   with self.subTest(condition=condition),tempfile.TemporaryDirectory() as d:
    root=Path(d);tools=root/'tools';tools.mkdir();code=tools/'module.py';code.write_bytes(b'implementation');before=code.read_bytes();parent=root/'parent.json';parent.write_text(json.dumps({'passed':condition!='failed_parent','files':supplement.manifest(tools)}));catalog=root/'catalog.json';catalog.write_bytes(b'{"models":[]}');digest=hashlib.sha256(catalog.read_bytes()).hexdigest()
    if condition=='catalog_changed':catalog.write_bytes(b'{"models":["unexpected"]}')
    if condition=='code_changed':code.write_bytes(b'changed')
    if condition=='existing_conflict':(tools/'astra_model_catalog.json').write_bytes(b'wrong')
    if condition=='unexpected_extra':(tools/'unexpected.json').write_bytes(b'unwanted')
    receipt=root/'receipt.json';args=['supplement','--tools',str(tools),'--parent-receipt',str(parent),'--catalog',str(catalog),'--receipt',str(receipt)]
    with patch.object(supplement,'CATALOG_SHA256',digest),patch.object(sys,'argv',args):
     if condition=='valid':
      supplement.main();self.assertEqual(code.read_bytes(),before);self.assertEqual((tools/'astra_model_catalog.json').read_bytes(),catalog.read_bytes());self.assertEqual((tools/'ovphysx-venv.provision.lock').read_bytes(),b'');self.assertTrue(json.loads(receipt.read_text())['passed'])
     else:
      with self.assertRaises(ValueError):supplement.main()
      self.assertFalse(receipt.exists())
      self.assertFalse((tools/'ovphysx-venv.provision.lock').exists())
 def test_staging_locator_is_derived_from_installer_and_confined(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);folder=root/'ovrtx-venv';folder.mkdir();cfg=folder/'pyvenv.cfg';stage=root/'.ovrtx-venv.staging-example';cfg.write_text('command = python -m venv '+str(stage)+'\n')
   self.assertEqual(staging_mapping(root),[(str(stage),'/tools/ovrtx-venv')])
   cfg.write_text('command = python -m venv /unrelated/staging\n')
   with self.assertRaises(ValueError):staging_mapping(root)
 def test_metadata_normalization_preserves_originals_native_hash_and_current_record(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);tools=root/'export';tools.mkdir();src=root/'repo/.venv/lib/python3.12/site-packages';src.mkdir(parents=True);dist=src/'example.dist-info';dist.mkdir();cache=dist/'uv_cache.json';cache.write_text('{"timestamp":{"secs_since_epoch":10}}');(src/'native.so').write_bytes(b'unchanged native bytes');(src/'editable.pth').write_text('/host/repo\n')
   def digest(data):return 'sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
   rows=[['native.so',digest((src/'native.so').read_bytes()),str((src/'native.so').stat().st_size)],['editable.pth',digest((src/'editable.pth').read_bytes()),str((src/'editable.pth').stat().st_size)],['example.dist-info/uv_cache.json',digest(cache.read_bytes()),str(cache.stat().st_size)],['example.dist-info/RECORD','','']];text=io.StringIO();csv.writer(text,lineterminator='\n').writerows(rows);(dist/'RECORD').write_text(text.getvalue());original_record=(dist/'RECORD').read_bytes()
   copy_tree(root/'repo/.venv',tools/'main-venv');editable=tools/'main-venv/lib/python3.12/site-packages/editable.pth';atomic_bytes(editable,b'/tools/repo\n')
   for name in ['ovphysx-venv','ovrtx-venv']:
    (root/name/'lib/python3.12/site-packages').mkdir(parents=True);(root/name/'bin').mkdir()
   (root/'ovrtx-venv/pyvenv.cfg').write_text('command = python -m venv '+str(root/'.ovrtx-venv.staging-fixture')+'\n')
   for name in ['activate','activate.csh','activate.fish']:(root/'ovrtx-venv/bin'/name).write_text('(.ovrtx-venv.staging-fixture)')
   for name in ['ovphysx-venv','ovrtx-venv']:copy_tree(root/name,tools/name)
   actions=normalize(root,tools,{'generated_relocations':[{'path':str(editable.relative_to(tools))}]})
   self.assertTrue(actions);self.assertEqual((dist/'RECORD').read_bytes(),original_record);self.assertTrue(cache.exists());self.assertFalse((tools/'main-venv/lib/python3.12/site-packages/example.dist-info/uv_cache.json').exists());self.assertEqual((tools/'main-venv/lib/python3.12/site-packages/native.so').read_bytes(),b'unchanged native bytes')
   actual=list(csv.reader(io.StringIO((tools/'main-venv/lib/python3.12/site-packages/example.dist-info/RECORD').read_text())));self.assertEqual(actual[0],rows[0]);self.assertEqual(actual[1][1],digest(editable.read_bytes()));self.assertFalse(any(r[0].endswith('uv_cache.json') for r in actual))
if __name__=='__main__':unittest.main()
