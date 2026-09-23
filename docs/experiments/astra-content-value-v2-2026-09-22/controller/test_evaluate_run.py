import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from evaluate_run import restore_workspace, safe_members, inspect_declaration, evaluation_command, sha256


class RestoreTests(unittest.TestCase):
    def archive(self, path, rows):
        with tarfile.open(path, 'w:gz') as tar:
            for name, kind, data in rows:
                member=tarfile.TarInfo(name)
                if kind=='link': member.type=tarfile.SYMTYPE;member.linkname=data;tar.addfile(member)
                else:
                    member.size=len(data);tar.addfile(member,io.BytesIO(data))

    def test_preserve_bytes_without_following_foreign_link_or_executing(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); out=root/'out';out.mkdir();archive=root/'input.tgz'
            self.archive(archive,[('nested/final.usd','file',b'unchanged'),('foreign','link','/etc/passwd'),
                                  ('never.py','file',b'raise AssertionError("Do not execute authored code")')])
            restore_workspace(archive,sha256(archive),out)
            self.assertEqual((out/'nested/final.usd').read_bytes(),b'unchanged')
            self.assertTrue((out/'foreign').is_symlink())
            self.assertEqual((out/'foreign').readlink(),Path('/etc/passwd'))

    def test_symlink_ancestor_refused_before_writes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'out';out.mkdir();archive=root/'input.tgz'
            self.archive(archive,[('escape','link','/tmp'),('escape/file','file',b'bad')])
            with self.assertRaises(ValueError):restore_workspace(archive,sha256(archive),out)
            self.assertEqual(list(out.iterdir()),[])

    def test_traversal_and_duplicate_members_refused(self):
        for rows in [[('../escape','file',b'bad')],[('x','file',b'1'),('x','file',b'2')]]:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);out=root/'out';out.mkdir();archive=root/'input.tgz';self.archive(archive,rows)
                with self.assertRaises(ValueError):restore_workspace(archive,sha256(archive),out)
                self.assertEqual(list(out.iterdir()),[])

    def test_transfer_digest_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'out';out.mkdir();archive=root/'input.tgz';self.archive(archive,[('x','file',b'ok')])
            with self.assertRaises(ValueError):restore_workspace(archive,'0'*64,out)

    def test_drawer_cannot_redirect_declared_scene(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);scene=root/'final.usd';scene.write_text('USD fixture');bindings=root/'bindings.json'
            b={'final_usd':'other.usd','drawer_body':'/Drawer','cabinet_body':'/Cabinet','drawer_joint':'/Joint','source_components':{}}
            bindings.write_text(json.dumps(b))
            delivery={'artifact_closure_valid':True,'task_outcome':'submitted','final_scene':'final.usd','bindings':'bindings.json',
                      'final_scene_sha256':sha256(scene),'bindings_sha256':sha256(bindings)}
            self.assertEqual(inspect_declaration(root,delivery,'01_drawer')['status'],'FAIL')
            b['final_usd']='final.usd';bindings.write_text(json.dumps(b));delivery['bindings_sha256']=sha256(bindings)
            self.assertIsNone(inspect_declaration(root,delivery,'01_drawer'))
            b['final_usd']='/work/final.usd';bindings.write_text(json.dumps(b));delivery['bindings_sha256']=sha256(bindings)
            self.assertIsNone(inspect_declaration(root,delivery,'01_drawer'))
            delivery['final_scene_sha256']='0'*64
            self.assertEqual(inspect_declaration(root,delivery,'01_drawer')['status'],'INCONCLUSIVE')

    def test_frozen_entrypoint_uses_declared_paths_and_original_source_alias(self):
        row={'case_id':'02_conveyor','argv':['old-python','old-evaluator','--usd','old','--bindings','old',
                '--inventory','old','--source-root','old','--output','old','--solver-python','old']}
        argv,cwd,report=evaluation_command('02_conveyor',{'final_scene':'result/final.usd','bindings':'result/bindings.json'}, {'cases':[row]})
        self.assertEqual(argv[argv.index('--usd')+1],'/work/result/final.usd')
        self.assertEqual(argv[argv.index('--source-root')+1],'/original')
        self.assertEqual(argv[argv.index('--solver-python')+1],'/tools/ovphysx-venv/bin/python')
        self.assertEqual(cwd,'/evaluator/cases/02_conveyor')
        self.assertEqual(report,'acceptance.json')


if __name__=='__main__':unittest.main()
