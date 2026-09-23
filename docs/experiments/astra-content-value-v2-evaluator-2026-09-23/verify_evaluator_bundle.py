"""Verify published membership without loading evaluator or authored code."""
import ast, hashlib, importlib.util, json, sys
from pathlib import Path
SCANNER_SHA='100339638100c2264c192710164d062f6adfdb99653b7dd48d7663b172ba6581'
EXCEPTIONS={'evaluator/qualification_plan.json': {'/native_host_authorization_required': True}, 'evaluator/reproduction/source_catalog.json': {'/cases/8/source/commit_message': 'Added assembly instructions'}, 'provenance/source_manifests/10_complex.json': {'/source/commit_message': 'Added assembly instructions'}}
EXCEPTION_HASHES={'evaluator/qualification_plan.json': '452f261e7f9b863eaa7d16020293395d6257e7f222c3209b8d09eb55532b7ee7', 'evaluator/reproduction/source_catalog.json': '22eca3849ee7e3a761f4d3540d67aae7ba82af2b739a74e4adaec2cd8f53c295', 'provenance/source_manifests/10_complex.json': '9239d7a5248e9e4e1c370fe027c68ac24369acf1f9081f21858bd41234489cd4'}
def h(data):return hashlib.sha256(data).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def verify(root):
    root=Path(root).resolve()
    scanner_path=root/'controller/publication.py'
    require(scanner_path.is_file() and not scanner_path.is_symlink() and h(scanner_path.read_bytes())==SCANNER_SHA,'Pinned scanner changed')
    spec=importlib.util.spec_from_file_location('verified_publication_scanner',scanner_path)
    scanner=importlib.util.module_from_spec(spec);spec.loader.exec_module(scanner)
    manifest_bytes=(root/'publication_manifest.json').read_bytes();manifest=json.loads(manifest_bytes)
    require(manifest['schema_version']=='evaluator-core-publication.v1' and manifest['geometry_payloads_included'] is False,'Unexpected bundle')
    expected=set();parsed=0
    for row in manifest['files']:
        rel=row['path'];p=Path(rel)
        require(not p.is_absolute() and '..' not in p.parts and p.as_posix()==rel and rel not in expected,'Invalid/duplicate member')
        expected.add(rel);file=root/p
        require(file.is_file() and not file.is_symlink() and file.resolve().is_relative_to(root),'Missing/escaping member')
        data=file.read_bytes();require(len(data)==row['bytes'] and h(data)==row['published_sha256'],'Published bytes changed: '+rel)
        if not row['projection']:require(row.get('original_retained_sha256') in (None,h(data)),'False exact-original assertion')
        if rel.endswith('.py'):ast.parse(data,filename=rel);parsed+=1
        screened=data
        if rel in EXCEPTIONS:
            require(h(data)==EXCEPTION_HASHES[rel],'Exact exception document changed')
            value=json.loads(data)
            for pointer,approved in EXCEPTIONS[rel].items():
                current=value;parts=pointer.strip('/').split('/')
                for part in parts[:-1]:current=current[int(part)] if isinstance(current,list) else current[part]
                require(type(current[parts[-1]]) is type(approved) and current[parts[-1]]==approved,'Exact public exception changed')
                del current[parts[-1]]
            screened=json.dumps(value).encode()
        scanner.scan(screened,rel.endswith('.json'),rel)
    scanner.scan(manifest_bytes,True,'publication_manifest.json')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    require(actual==expected|{'publication_manifest.json'},'Missing/unlisted member')
    return {'passed':True,'files':len(actual),'python_syntax_files':parsed,'manifest_sha256':h(manifest_bytes),'geometry_payloads_included':False,'scope':'Published byte identity/privacy/syntax only; no physics or outcome inference.'}
if __name__=='__main__':print(json.dumps(verify(sys.argv[1] if len(sys.argv)>1 else '.'),indent=2))
