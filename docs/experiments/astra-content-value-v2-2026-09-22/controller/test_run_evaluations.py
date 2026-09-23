import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import run_evaluations as batch


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_archive(path, run, files):
    rows = [{'path': name, 'kind': 'file', 'mode': 0o600, 'bytes': len(data),
             'sha256': batch.hashlib.sha256(data).hexdigest()} for name, data in files.items()]
    manifest = json.dumps({'run_id': run, 'files': rows}, sort_keys=True).encode()
    with tarfile.open(path, 'w:gz') as tar:
        for name, data in dict(files, **{'private/retention_manifest.json': manifest}).items():
            member = tarfile.TarInfo(run + '/' + name); member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    return {'run_id': run, 'archive_sha256': batch.sha256(path), 'archive_bytes': path.stat().st_size,
            'manifest_sha256': batch.hashlib.sha256(manifest).hexdigest(), 'local_member_verification': True}


class GateTests(unittest.TestCase):
    def fixture(self, root):
        protocol = {'lane_assignment': {}, 'lanes': [{'id': 'lane' + str(i)} for i in range(4)]}
        jobs = []
        for n in range(1, 11):
            case = f'{n:02d}_fixture'; lane = 'lane' + str((n - 1) % 4)
            protocol['lane_assignment'][case] = lane
            arms = ['plain_astra', 'content_agents']
            if n % 2 == 0: arms.reverse()
            for arm in arms:
                jobs.append({'run_id': case + '_' + arm, 'case_id': case, 'arm': arm, 'lane_id': lane,
                             'task_sha256': 'a' * 64, 'input_sha256': 'b' * 64,
                             'source_sha256': 'c' * 64, 'protocol_sha256': 'd' * 64})
        cfg = {'output': str(root / 'dispatch'), 'retained_archives': str(root / 'archives'),
               'database': str(root / 'ledger.sqlite'), 'protocol_file': str(root / 'protocol.json')}
        write(root / 'protocol.json', protocol); write(root / 'jobs.json', {'jobs': jobs})
        write(root / 'dispatch/queue_result.json', {'protocol_sha256': batch.sha256(root / 'protocol.json'),
              'jobs_sha256': batch.sha256(root / 'jobs.json'),
              'lanes': [[{'run_id': j['run_id']} for j in jobs if j['lane_id'] == lane['id']] for lane in protocol['lanes']]})
        connection = sqlite3.connect(cfg['database'])
        connection.executescript('CREATE TABLE settings(key TEXT,value TEXT); CREATE TABLE runs(id TEXT,status TEXT,lane TEXT,lease TEXT,reap_sha TEXT,spec_sha TEXT);')
        connection.execute('INSERT INTO settings VALUES(?,?)', ('protocol', json.dumps(protocol)))
        for job in jobs:
            run = job['run_id']; receipt = {'run_id': run, 'lease': 'lease-' + run, 'cgroup_populated': 0, 'namespace_init_exited': True}
            connection.execute('INSERT INTO runs VALUES(?,?,?,?,?,?)', (run, 'reaped', job['lane_id'], receipt['lease'], batch.object_hash(receipt), batch.object_hash(job)))
            write(root / 'dispatch' / run / 'worker_reap.json', receipt)
            directory = root / 'archives' / run; directory.mkdir(parents=True)
            proof = make_archive(directory / 'complete.tgz', run, {
                'private/author_delivery.json': json.dumps({'task_outcome': 'not_submitted', 'marker': 'exact supervisor declaration'}).encode(),
                'home/private.txt': b'PRIVATE_HOME_NEVER_RESTORED',
                'workspace/final.usda': b'#usda 1.0\n',
                'workspace/never_execute.py': b'raise RuntimeError("AUTHORED_CODE_MUST_NEVER_EXECUTE")'})
            write(directory / 'verification.json', proof)
            write(directory / 'worker_cleanup.json', {'run_id': run, 'removed_only': ['workspace', 'home'], 'verified_archive_sha256': proof['archive_sha256']})
            write(root / 'dispatch' / run / 'retention.json', {'run_id': run, 'verified': True, 'archive_sha256': proof['archive_sha256']})
        connection.commit(); connection.close()
        return cfg, jobs, protocol, root / 'jobs.json'

    def test_all_twenty_required_and_exact_private_declaration(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d)); result = batch.author_gate(*args)
            self.assertEqual(len(result), 20)
            self.assertTrue(all(x['declaration']['marker'] == 'exact supervisor declaration' for x in result))

    def test_one_unreaped_author_blocks_whole_batch(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d)); con = sqlite3.connect(args[0]['database'])
            con.execute("UPDATE runs SET status='running' WHERE id=?", (args[1][-1]['run_id'],)); con.commit(); con.close()
            with self.assertRaisesRegex(ValueError, 'not reaped'): batch.author_gate(*args)

    def test_incomplete_queue_and_changed_job_refused(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d)); p = Path(args[0]['output']) / 'queue_result.json'; value = batch.read(p)
            value['lanes'][0].pop(); write(p, value)
            with self.assertRaisesRegex(ValueError, 'not complete'): batch.author_gate(*args)
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d)); args[1][0]['unexpected_source_override'] = True
            with self.assertRaisesRegex(ValueError, 'specification differs'): batch.author_gate(*args)

    def test_corrupt_retention_blocks_whole_batch(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.fixture(Path(d)); p = Path(args[0]['retained_archives']) / args[1][-1]['run_id'] / 'complete.tgz'
            p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaises(AssertionError): batch.author_gate(*args)

    def test_workspace_package_excludes_home_and_private_without_execution(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); args = self.fixture(root); job = batch.author_gate(*args)[0]
            result = batch.pack_workspace(job['archive'], job['proof'], job['run_id'], root / 'workspace.tgz')
            with tarfile.open(root / 'workspace.tgz') as tar:
                batch.safe_members(tar)
                self.assertEqual(set(tar.getnames()), {'final.usda', 'never_execute.py'})
                self.assertIn(b'AUTHORED_CODE_MUST_NEVER_EXECUTE', tar.extractfile('never_execute.py').read())
            self.assertEqual(result['original_retention_sha256'], job['proof']['archive_sha256'])


class ContractTests(unittest.TestCase):
    def test_config_forces_evaluator_rootfs_without_inheriting_credentials(self):
        cfg = batch.evaluator_config({'rootfs': '/wrong-author-root', 'tools': '/tools', 'lanes': {},
                                      'system_mounts': [], 'auth_file': '/secret'}, '/opt/experiment', 'fresh')
        self.assertEqual(cfg['rootfs'], '/opt/experiment/evaluator_runtime/rootfs')
        self.assertNotIn('auth_file', cfg)

    def test_exact_declaration_and_original_ancestor_not_authored_source(self):
        job = {'run_id': 'author', 'case_id': '09_printer', 'arm': 'plain_astra', 'source': '/opt/experiment/assets/09_printer/extracted',
               'declaration': {'final_scene': 'chosen/final.usd'}, 'proof': {'archive_sha256': 'a' * 64}}
        invocations = {'cases': [{'case_id': '09_printer', 'source_root': '/opt/experiment/assets/09_printer/extracted'}]}
        result = batch.evaluation_request(job, 'fresh', {'id': 'lane0'}, '/opt/experiment', invocations, 'b' * 64)
        self.assertIs(result['declaration'], job['declaration'])
        self.assertEqual(result['original'], job['source'])
        invocations['cases'][0]['source_root'] = '/opt/experiment/assets/08_other'
        with self.assertRaisesRegex(ValueError, 'outside case'): batch.evaluation_request(job, 'fresh', {}, '/opt/experiment', invocations, 'b' * 64)

    def test_early_rejection_retention_does_not_require_launch_json(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); out = root / 'runs/eval'; stage = root / 'stage'; stage.mkdir()
            request = {'run_id': 'eval', 'lane': {'id': 'lane0'}}; write(stage / 'request.json', request)
            write(out / 'private/evaluation.json', {'status': 'FAIL', 'run_id': 'eval'})
            receipt = {'run_id': 'eval', 'lane_id': 'lane0', 'cgroup_populated': 0, 'terminal': True,
                       'evaluation_sha256': batch.sha256(out / 'private/evaluation.json'), 'request_sha256': batch.sha256(stage / 'request.json')}
            write(out / 'private/batch_evaluation.json', receipt)
            loaded = (stage, {'runs_root': str(root / 'runs')}, request, {}, root / 'group')
            with patch.object(batch, 'worker_load', return_value=loaded), patch.object(batch, 'populated', return_value=0), patch.object(batch, 'require_root_directory'):
                self.assertEqual(batch.worker_terminal(stage)[-1], receipt)
            self.assertFalse((out / 'private/launch.json').exists())

    def test_occupied_lane_prevents_retention(self):
        with patch.object(batch, 'worker_load', return_value=(None, {}, {}, {}, Path('/group'))), patch.object(batch, 'populated', return_value=1):
            with self.assertRaisesRegex(ValueError, 'still populated'): batch.worker_terminal('/stage')

    def test_wrong_rootfs_and_oversize_stage_fail_before_mount(self):
        with patch.object(batch.os, 'geteuid', return_value=0), patch.object(batch.subprocess, 'run') as spawn:
            with self.assertRaisesRegex(ValueError, 'Wrong evaluator rootfs'): batch.worker_prepare({'config': {'rootfs': '/wrong'}, 'request': {}})
            spawn.assert_not_called()
        cfg = {'rootfs': str(batch.ROOT / 'evaluator_runtime/rootfs'), 'runs_root': str(batch.ROOT / 'evaluation_runs/fresh'), 'lanes': {'l0': {'cgroup': '/group'}}}
        req = {'run_id': 'eval', 'lane': {'id': 'l0', 'memory_bytes': 32 * 2**30}}
        with patch.object(batch.os, 'geteuid', return_value=0), patch.object(batch, 'populated', return_value=0), patch.object(batch, 'check_limits'), patch.object(batch.subprocess, 'run') as spawn:
            with self.assertRaisesRegex(ValueError, 'staging cap'): batch.worker_prepare({'config': cfg, 'request': req, 'archive_bytes': batch.MAX_STAGE_BYTES + 1})
            spawn.assert_not_called()

    def test_existing_atomic_lane_lease_prevents_concurrent_staging(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); write(root / 'freeze.json', {'fixture': True})
            lease = root / 'evaluation_lane_leases/l0'; lease.mkdir(parents=True)
            write(lease / 'lease.json', {'run_id': 'previous_unretained'})
            cfg = {'rootfs': str(root / 'evaluator_runtime/rootfs'), 'runs_root': str(root / 'evaluation_runs/fresh'), 'lanes': {'l0': {'cgroup': '/group'}}}
            req = {'run_id': 'eval', 'lane': {'id': 'l0', 'memory_bytes': 32 * 2**30}, 'freeze_file': str(root / 'freeze.json'),
                   'freeze_sha256': batch.sha256(root / 'freeze.json'), 'workspace_archive': str(root / 'evaluation_staging/eval/payload/workspace.tgz')}
            with patch.object(batch, 'ROOT', root), patch.object(batch.os, 'geteuid', return_value=0), patch.object(batch, 'populated', return_value=0), patch.object(batch, 'check_limits'), patch.object(batch, 'enter_lane'), patch.object(batch, 'verify_freeze'), patch.object(batch, 'verify_original_inputs'), patch.object(batch.subprocess, 'run') as spawn:
                with self.assertRaises(FileExistsError): batch.worker_prepare({'config': cfg, 'request': req, 'archive_bytes': 1024})
                spawn.assert_not_called()
            self.assertEqual(batch.read(lease / 'lease.json')['run_id'], 'previous_unretained')

    def test_original_bytes_and_frozen_ancestor_are_both_required(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root / 'assets/09_printer/extracted'; source.mkdir(parents=True)
            (source / 'original.step').write_bytes(b'original source')
            write(root / 'protocol/author_source_closures.json', {'cases': [{'case_id': '09_printer', 'source_root': 'assets/09_printer/extracted',
                  'files': [{'path': 'original.step', 'size': 15, 'sha256': batch.sha256(source / 'original.step')}]}]})
            write(root / 'invocations.json', {'cases': [{'case_id': '09_printer', 'source_root': str(source)}]})
            request = {'case_id': '09_printer', 'source': str(source), 'original': str(source), 'invocations_file': str(root / 'invocations.json')}
            with patch.object(batch, 'ROOT', root):
                batch.verify_original_inputs(request)
                with self.assertRaisesRegex(ValueError, 'ancestor differs'): batch.verify_original_inputs(dict(request, original=str(root / 'author_copy')))
                (source / 'original.step').write_bytes(b'changed!')
                with self.assertRaisesRegex(ValueError, 'source bytes changed'): batch.verify_original_inputs(request)

    def test_frozen_reference_drift_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); reference = root / 'evaluator/reference.npz'; reference.parent.mkdir()
            reference.write_bytes(b'reference')
            write(root / 'freeze.json', {'frozen': True, 'status': 'PASS', 'files': [{'path': 'evaluator/reference.npz', 'sha256': batch.sha256(reference)}]})
            batch.verify_freeze(root / 'freeze.json', root)
            reference.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Frozen file changed'): batch.verify_freeze(root / 'freeze.json', root)


class RetentionSequenceTests(unittest.TestCase):
    def fixture(self, root, corrupt=False):
        terminal = {'run_id': 'eval', 'lane_id': 'lane0', 'terminal': True, 'cgroup_populated': 0, 'status': 'FAIL'}
        source = root / 'remote.tgz'
        proof = make_archive(source, 'eval', {'private/batch_evaluation.json': json.dumps(terminal).encode(),
                             'private/evaluation.json': b'{"status":"FAIL"}', 'home/evidence/rejection.json': b'{}', 'workspace/final.usda': b'#usda 1.0'})
        calls = []
        def remote(mode, **kwargs):
            calls.append(mode)
            if mode == 'export': kwargs['stdout'].write(source.read_bytes()[:-1] if corrupt else source.read_bytes()); return subprocess.CompletedProcess([], 0)
            if mode == 'receipt': return subprocess.CompletedProcess([], 0, json.dumps(proof))
            if mode == 'cleanup':
                local_proof = json.loads(kwargs['input']); self.assertTrue(local_proof['local_member_verification'])
                self.assertTrue((root / 'retained/verification.json').exists())
                self.assertTrue((root / 'retained/complete.tgz').exists())
                return subprocess.CompletedProcess([], 0, json.dumps({'run_id': 'eval', 'verified_archive_sha256': proof['archive_sha256']}))
            raise AssertionError(mode)
        return terminal, remote, calls

    def test_complete_verified_archive_precedes_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); terminal, remote, calls = self.fixture(root)
            result = batch.retain_evaluation(remote, root / 'retained', 'eval', terminal)
            self.assertTrue(result['verified']); self.assertEqual(calls, ['export', 'receipt', 'cleanup'])
            self.assertEqual(os.stat(root / 'retained/verification.json').st_mode & 0o777, 0o600)

    def test_corrupt_export_never_cleans_worker(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); terminal, remote, calls = self.fixture(root, corrupt=True)
            with self.assertRaises(AssertionError): batch.retain_evaluation(remote, root / 'retained', 'eval', terminal)
            self.assertEqual(calls, ['export', 'receipt'])
            self.assertTrue((root / 'retained/complete.partial.tgz').exists())


if __name__ == '__main__': unittest.main()
