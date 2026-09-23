import copy
import json
import tempfile
import unittest
from pathlib import Path
import publication as p

class PublicationTests(unittest.TestCase):
    def fixture(self, root):
        cases=[f'{i:02}_synthetic' for i in range(1,11)]
        data={'protocol/frozen_protocol.json':{'frozen':True,'model':'gpt-6-astra','reasoning_effort':'ultra','gpu_uuid':'GPU-'+'12345678-1234-1234-1234-123456789012'},
              'controller/private/model_catalog.json':{'base_instructions':'private base instruction sentinel'},
              'evidence/qualification.json':{'schema_version':'synthetic','passed':True,'checks':{'physical_gpu':True},'messages':[{'content':'raw model sentinel'}],'child_session_id':'private-session-sentinel'}}
        for case in cases:
            data[f'protocol/public_inputs/{case}/task.json']={'case_id':case,'task':'synthetic task'}
            data[f'protocol/public_inputs/{case}/{case}_source_inventory.json']={'source_sha256':'a'*64}
        for rel,value in data.items():
            path=root/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(p.encode(value))
        for rel,content in {'controller/publication.py':Path(p.__file__).read_bytes(),'environment/README.md':b'Qualified synthetic environment fixture.\n','environment/pins.json':b'{"synthetic":true}\n'}.items():
            path=root/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content);data[rel]=None
        freeze={'schema_version':'astra-content-value-freeze.v2','frozen':True,'status':'PASS','created_utc':'2026-09-22T00:00:00Z','qualified_cases':cases,'protocol_file':'protocol/frozen_protocol.json','public_inputs_directory':'protocol/public_inputs','files':[{'path':rel,'sha256':p.digest((root/rel).read_bytes())}for rel in sorted(data)],'qualification_evidence':[{'path':'evidence/qualification.json','sha256':p.digest((root/'evidence/qualification.json').read_bytes())}]}
        path=root/'protocol/freeze.json';path.write_bytes(p.encode(freeze));return path

    def test_deterministic_create_only_excludes_private_and_binds_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'source';root.mkdir();freeze=self.fixture(root)
            value=json.loads(freeze.read_bytes());private=next(x for x in value['files']if x['path']=='controller/private/model_catalog.json')
            value['qualification_evidence'].append(private);freeze.write_bytes(p.encode(value))
            a=Path(tmp)/'a';b=Path(tmp)/'b'
            ra=p.build(root,freeze,a);rb=p.build(root,freeze,b)
            self.assertEqual(ra,rb)
            self.assertFalse((a/'controller/private/model_catalog.json').exists())
            contents=b'\n'.join(x.read_bytes()for x in a.rglob('*')if x.is_file())
            for secret in (b'private base instruction sentinel',b'raw model sentinel',b'private-session-sentinel',b'GPU-'+b'12345678'):
                self.assertNotIn(secret,contents)
            manifest=json.loads((a/'publication_manifest.json').read_bytes())
            row=next(x for x in manifest['files']if x['path']=='protocol/frozen_protocol.json')
            self.assertTrue(row['projection']);self.assertNotEqual(row['published_sha256'],row['original_retained_sha256'])
            task=next(x for x in manifest['files']if x['path'].endswith('/task.json'))
            self.assertFalse(task['projection']);self.assertEqual(task['published_sha256'],task['original_retained_sha256'])
            with self.assertRaises(ValueError):p.build(root,freeze,a)

    def test_changed_frozen_input_leaves_no_partial_public_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'source';root.mkdir();freeze=self.fixture(root)
            (root/'protocol/public_inputs/01_synthetic/task.json').write_text('{}')
            out=Path(tmp)/'public'
            with self.assertRaises(ValueError):p.build(root,freeze,out)
            self.assertFalse(out.exists())

    def test_unfrozen_or_incomplete_cases_rejected(self):
        for change in ({'frozen':False},{'status':'PENDING'},{'qualified_cases':['01_synthetic']}):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);freeze=self.fixture(root);value=json.loads(freeze.read_bytes());value.update(change);freeze.write_bytes(p.encode(value))
                with self.assertRaises(ValueError):p.build(root,freeze,root/'public')

    def test_secret_or_private_field_in_exact_task_rejected(self):
        for value in ({'api_key':'real-secret'},{'task':'Bearer '+'abcdefghijklmnopqrstuvwxyz'},{'task':'/Users/'+'private-person/secret'}):
            with self.subTest(value=value),self.assertRaises(ValueError):p.scan(p.encode(value))

    def test_projection_never_passes_raw_auth_value(self):
        value={'passed':True,'messages':[{'content':'hidden'}],'checks':{'api_key':'hidden','ok':True},'session_id':'hidden'}
        self.assertEqual(p.project(value,True),{'passed':True,'checks':{'ok':True}})
        with self.assertRaises(ValueError):p.project({'reason':'Bearer '+'abcdefghijklmnopqrstuvwxyz'},True)

    def test_nested_raw_fields_removed_but_model_effort_retained(self):
        value={'reasoning_effort':'ultra','wire_reasoning_effort':'xhigh','checks':{
          'passed':True,'content':'private tool body','arguments':{'cmd':'private command'},
          'stdout':'private output','reasoning':'private reasoning','transcript':'private transcript'}}
        self.assertEqual(p.project(value,True),{'reasoning_effort':'ultra','wire_reasoning_effort':'xhigh','checks':{'passed':True}})

    def test_exact_no_secret_provider_exception_is_digest_and_path_bound(self):
        provider=Path(p.__file__).parents[1]/'environment/native_provider.json'
        data=provider.read_bytes()
        p.scan(data,True,'environment/native_provider.json')
        with self.assertRaises(ValueError):p.scan(data,True,'evidence/arbitrary_provider.json')
        changed=json.loads(data);changed['model_providers']['trial_gateway']['base_url']='https://example.com/v1'
        with self.assertRaises(ValueError):p.scan(p.encode(changed),True,'environment/native_provider.json')
        changed=json.loads(data);changed['extra_secret']='private-value'
        with self.assertRaises(ValueError):p.scan(p.encode(changed),True,'environment/native_provider.json')

    def test_corruption_or_unlisted_file_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'source';root.mkdir();freeze=self.fixture(root);out=Path(tmp)/'public';p.build(root,freeze,out)
            extra=out/'raw.json';extra.write_text('{}')
            with self.assertRaises(ValueError):p.verify(out)
            extra.unlink();(out/'README.md').write_text('changed')
            with self.assertRaises(ValueError):p.verify(out)

    def test_unsafe_freeze_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);freeze=self.fixture(root);task=root/'protocol/public_inputs/01_synthetic/task.json';task.unlink();task.symlink_to('/etc/hosts')
            with self.assertRaises(ValueError):p.build(root,freeze,root/'public')
        for name in ('/etc/passwd','../escape','controller/private/key.json','evidence/foo.private.json','evidence/raw.tgz'):
            with self.subTest(name=name),self.assertRaises(ValueError):p.safe_relative(name)

    def test_narrow_source_literal_exceptions(self):
        p.scan(b'Bearer '+b'synthetic-token',False,'harness/tests/test_harness.py')
        with self.assertRaises(ValueError):p.scan(b'Bearer '+b'abcdefghijklmnopqrstuvwxyz',False,'harness/tests/test_harness.py')
        with self.assertRaises(ValueError):p.scan(b'Bearer '+b'synthetic-token',False,'environment/README.md')
        p.scan(b'interpreter/home/'+b'version entries',False,'environment/export_namespace_tools.py')
        with self.assertRaises(ValueError):p.scan(b'/home/'+b'actual-user/private',False,'environment/export_namespace_tools.py')
        p.scan(b'/home/'+b'evaluator',False,'controller/evaluate_run.py')
        with self.assertRaises(ValueError):p.scan(b'/home/'+b'actual-user/private',False,'controller/evaluate_run.py')
        p.scan(b'Bearer '+b'synthetic-health-invalid-token',False,'harness/network_health.py')
        with self.assertRaises(ValueError):p.scan(b'Bearer '+b'abcdefghijklmnopqrstuvwxyz',False,'harness/network_health.py')

    def test_manual_capture_caveats_survive_metadata_projection(self):
        raw={'status':'PASS','scope':'Manual interrupted child only','limits':['Not automatic full-run capture'],'automatic_capture_qualified':False,'whole_q13_ui_complete':False,'downstream_disconnected':True,'timing_inference_used':False,'association':{'complete':True,'terminal_billing_complete':False,'admission_identity_associations':1,'thread_id':'private-thread'}}
        value=p.project(raw,metadata_only=True)
        for name in ('scope','limits','automatic_capture_qualified','whole_q13_ui_complete','downstream_disconnected','timing_inference_used'):self.assertEqual(value[name],raw[name])
        self.assertTrue(value['association']['complete']);self.assertFalse(value['association']['terminal_billing_complete'])
        self.assertNotIn('private-thread',p.encode(value).decode());p.scan(p.encode(value))

    def test_ui_association_counts_preserved_without_session_identifiers(self):
        value={'complete':False,'terminal_billing_complete':False,'terminal_response_id_associations':2,'admission_identity_associations':3,'timing_inference_used':False,'requests':[{'thread_id':'private-thread','turn_id':'private-turn','session_id':'private-session','associated':True}],'unledgered_response_ids':['private-response'],'uncovered_threads':['private-thread']}
        projected=p.project(value,metadata_only=True)
        self.assertEqual(projected['terminal_response_id_associations'],2)
        self.assertEqual(projected['admission_identity_associations'],3)
        self.assertFalse(projected['complete']);self.assertFalse(projected['terminal_billing_complete'])
        self.assertNotIn('private-',p.encode(projected).decode());p.scan(p.encode(projected))
        self.assertNotIn('terminal_response_id_associations',p.project({'terminal_response_id_associations':'private-response'}))
        with self.assertRaises(ValueError):p.scan(p.encode({'terminal_response_id_associations':'private-response'}))

    def test_manifest_private_field_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'source';root.mkdir();freeze=self.fixture(root);out=Path(tmp)/'public';p.build(root,freeze,out)
            m=out/'publication_manifest.json';value=json.loads(m.read_bytes());value['authorization']='private-value';m.write_bytes(p.encode(value))
            with self.assertRaises(ValueError):p.verify(out)

    def test_capture_qualification_preserves_failure_scope_and_typed_counts(self):
        raw={'status':'PASS','native_task_verdict':'fail','task_outcome':'Expected task FAIL',
             'sessions':2,'automatic_sessions':2,'ui_gateway_response_ids_reconciled':True,
             'terminal_response_id_and_admission_associations':15,'automatic_capture_qualified':True}
        self.assertEqual(p.project(raw,True),raw)
        p.scan(p.encode(p.project(raw,True)))
        for value in ('private-response',['private-response'],{'session_id':'private-session'}):
            projected=p.project({'sessions':value,'ui_gateway_response_ids_reconciled':value,
                                 'terminal_response_id_and_admission_associations':value},True)
            self.assertEqual(projected,{})
            with self.assertRaises(ValueError):p.scan(p.encode({'ui_gateway_response_ids_reconciled':value}))

    def test_pin_credentials_rejected_before_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'source';root.mkdir();freeze=self.fixture(root)
            pins=root/'environment/pins.json';pins.write_bytes(p.encode({'api_key':'private-value'}))
            f=json.loads(freeze.read_bytes())
            next(x for x in f['files']if x['path']=='environment/pins.json')['sha256']=p.digest(pins.read_bytes())
            freeze.write_bytes(p.encode(f));out=Path(tmp)/'public'
            with self.assertRaises(ValueError):p.build(root,freeze,out)
            self.assertFalse(out.exists())

if __name__=='__main__':unittest.main()
