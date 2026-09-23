import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from retain_run import verify_archive
from worker_retention import cleanup


class RetentionTests(unittest.TestCase):
    def prepare(self, directory):
        root = Path(directory)/'qrun'
        for name in ('workspace', 'home', 'private'):
            (root/name).mkdir(parents=True)
        (root/'workspace/result.usda').write_text('test fixture, not USD')
        (root/'home/session.jsonl').write_text('{"model":"gpt-6-astra","effort":"ultra"}\n')
        (Path(directory)/'foreign.txt').write_text('FOREIGN_CONTENT_MUST_NOT_BE_READ')
        (root/'workspace/foreign_link').symlink_to(Path(directory)/'foreign.txt')
        os.mkfifo(root/'workspace/pipe')
        archive = Path(directory)/'export.tgz'
        self.export(root, archive)
        receipt = json.loads((root/'private/archive_receipt.json').read_text())
        return root, archive, verify_archive(archive, 'qrun', receipt)

    def export(self, root, archive):
        with archive.open('wb') as f:
            subprocess.run([sys.executable, '-c',
                'from pathlib import Path; import sys; from worker_retention import export; export(Path(sys.argv[1]), "qrun")',
                str(root)], cwd=Path(__file__).parent, stdout=f, check=True, timeout=10)

    def test_complete_bytes_links_and_special_file_handling(self):
        with tempfile.TemporaryDirectory() as d:
            root, archive, proof = self.prepare(d)
            self.assertTrue(proof['local_member_verification'])
            with tarfile.open(archive) as tar:
                link = tar.getmember('qrun/workspace/foreign_link')
                self.assertTrue(link.issym())
                self.assertNotIn('qrun/workspace/pipe', tar.getnames())
                text = b''.join(tar.extractfile(x).read() for x in tar if x.isfile())
                self.assertNotIn(b'FOREIGN_CONTENT_MUST_NOT_BE_READ', text)
            with contextlib.redirect_stdout(io.StringIO()):
                cleanup(root, 'qrun', proof)
                cleanup(root, 'qrun', proof)
            self.assertFalse((root/'workspace').exists())
            self.assertTrue((root/'private/archive_receipt.json').is_file())
            self.assertEqual((Path(d)/'foreign.txt').read_text(), 'FOREIGN_CONTENT_MUST_NOT_BE_READ')

    def test_reexport_is_identical(self):
        with tempfile.TemporaryDirectory() as d:
            root, archive, proof = self.prepare(d)
            second = Path(d)/'second.tgz'
            self.export(root, second)
            self.assertEqual(archive.read_bytes(), second.read_bytes())

    def test_corrupt_archive_rejected_before_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            root, archive, proof = self.prepare(d)
            archive.write_bytes(archive.read_bytes()[:-5])
            with self.assertRaises(AssertionError):
                verify_archive(archive, 'qrun', proof)
            self.assertTrue((root/'workspace/result.usda').exists())

    def test_changed_run_is_retained(self):
        with tempfile.TemporaryDirectory() as d:
            root, archive, proof = self.prepare(d)
            (root/'workspace/result.usda').write_text('changed')
            with self.assertRaises(ValueError):
                cleanup(root, 'qrun', proof)
            self.assertTrue((root/'workspace/result.usda').exists())

    def test_unverified_proof_refuses_deletion(self):
        with tempfile.TemporaryDirectory() as d:
            root, archive, proof = self.prepare(d)
            proof['local_member_verification'] = False
            with self.assertRaises(ValueError):
                cleanup(root, 'qrun', proof)
            self.assertTrue((root/'home/session.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
