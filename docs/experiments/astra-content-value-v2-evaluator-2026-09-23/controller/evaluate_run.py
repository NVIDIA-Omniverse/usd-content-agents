"""Restore immutable author data and evaluate it in a credential-free namespace.

No authored program is executed. The evaluator sees the original /work and
/source aliases read-only; only /evidence and temporary files are writable.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import resource
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'harness'))
from common import closed_file, identifier, read, require, sha256, write_new
from isolation import check_host_owned, check_limits, kill_and_reap, populated
from run_arm import mount_run_storage


def pack_workspace(archive, proof, run_id, destination):
    """Select workspace bytes from an already verified complete private archive."""
    from retain_run import verify_archive
    verify_archive(Path(archive), run_id, proof)
    prefix = run_id+'/workspace/'
    rows = []
    with tarfile.open(archive, 'r:gz') as source, tarfile.open(destination, 'x:gz') as target:
        for member in source:
            if not member.name.startswith(prefix):
                continue
            name = member.name[len(prefix):]
            if not name:
                continue
            info = tarfile.TarInfo(name)
            info.mode, info.mtime, info.type = member.mode, 0, member.type
            info.linkname, info.size = member.linkname, member.size
            target.addfile(info, source.extractfile(member) if member.isfile() else None)
            rows.append({'path': name, 'kind': 'file' if member.isfile() else 'link' if member.issym() else 'directory'})
    return {'run_id': run_id, 'original_retention_sha256': proof['archive_sha256'],
            'workspace_archive_sha256': sha256(destination), 'members': rows,
            'scope': 'Workspace only; no home, auth, session history or authored execution.'}


def safe_members(tar, maximum=8*2**30):
    members = tar.getmembers()
    names = set()
    links = set()
    for member in members:
        path = PurePosixPath(member.name)
        require(member.name and not path.is_absolute() and '..' not in path.parts and str(path) == member.name,
                'Noncanonical archive path')
        require(member.name not in names, 'Duplicate archive path')
        require(member.isfile() or member.isdir() or member.issym(), 'Unsupported archive member')
        names.add(member.name)
        if member.issym(): links.add(member.name)
    require(sum(m.size for m in members if m.isfile()) <= maximum, 'Workspace exceeds frozen capacity')
    for member in members:
        require(not any(str(p) in links for p in PurePosixPath(member.name).parents), 'Symlink ancestor in archive')
    return members


def restore_workspace(archive, expected_sha, destination):
    require(sha256(archive) == expected_sha, 'Workspace transfer changed')
    destination = Path(destination)
    require(destination.is_dir() and not any(destination.iterdir()), 'Fresh empty workspace required')
    with tarfile.open(archive, 'r:gz') as tar:
        members = safe_members(tar)
        # Never follow a link while creating data. Links are restored last as metadata.
        for member in sorted(members, key=lambda m: (m.issym(), len(PurePosixPath(m.name).parts), m.name)):
            target = destination/member.name
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            if member.isdir(): target.mkdir(exist_ok=True, mode=0o755)
            elif member.issym(): target.symlink_to(member.linkname)
            else:
                with target.open('xb') as out, tar.extractfile(member) as src:
                    shutil.copyfileobj(src, out, 1024*1024)
                target.chmod(0o444)
    destination.chmod(0o755)


def inspect_declaration(workspace, declaration, case_id):
    """Only inspect JSON and file identity; never import authored code."""
    if declaration.get('artifact_closure_valid') is not True or declaration.get('task_outcome') != 'submitted':
        return {'status': 'FAIL', 'reason': 'Required immutable scene/bindings were not submitted.',
                'delivery': declaration}
    try:
        scene = closed_file(workspace, declaration['final_scene'])
        bindings = closed_file(workspace, declaration['bindings'])
        if sha256(scene) != declaration['final_scene_sha256'] or sha256(bindings) != declaration['bindings_sha256']:
            return {'status':'INCONCLUSIVE','retention_integrity':False,'reason':'Retained declared file digest differs'}
        raw = read(bindings)
        require(isinstance(raw, dict), 'Bindings must be a JSON object')
        if case_id == '01_drawer':
            require(isinstance(raw.get('final_usd'), str), 'Drawer bindings need final_usd')
            for key in ('drawer_body','cabinet_body','drawer_joint'):
                require(isinstance(raw.get(key),str) and raw[key].startswith('/'), 'Missing absolute drawer prim binding: '+key)
            mapping=raw.get('source_components')
            require(isinstance(mapping,dict) and all(isinstance(k,str) and isinstance(v,list)
                    and all(isinstance(p,str) and p.startswith('/') for p in v) for k,v in mapping.items()),
                    'Invalid drawer source_components mapping')
            value=raw['final_usd']
            if value.startswith('/work/'):
                named=Path(workspace)/value.removeprefix('/work/')
            else:
                require(not Path(value).is_absolute(), 'Drawer USD binding outside /work')
                named=bindings.parent/value
            require(named.resolve() == scene.resolve(),
                    'Drawer binding must name the declared final scene')
        return None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {'status': 'FAIL', 'reason': 'Concrete submission rejection: '+str(exc)}


def evaluation_command(case_id, declaration, invocations):
    scene = '/work/'+declaration['final_scene']
    bindings = '/work/'+declaration['bindings']
    if case_id == '01_drawer':
        return ['/tools/main-venv/bin/python', '-B', '/evaluator/drawer/drawer_evaluate.py',
                '--bindings', bindings, '--output', '/evidence/result',
                '--solver', '/tools/ovphysx-venv/bin/python'], '/evaluator/drawer', 'report.json'
    row = next(x for x in invocations['cases'] if x['case_id'] == case_id)
    argv = list(row['argv'])
    argv[0] = '/tools/main-venv/bin/python'
    argv[1] = '/evaluator/cases/'+case_id+'/evaluate.py'
    replacements = {'--usd': scene, '--bindings': bindings, '--inventory': '/evaluator/cases/'+case_id+'/reference/source_inventory.json',
                    '--source-root': '/original', '--output': '/evidence/result', '--solver-python': '/tools/ovphysx-venv/bin/python'}
    for flag, value in replacements.items():
        require(flag in argv, 'Missing frozen evaluator argument '+flag)
        argv[argv.index(flag)+1] = value
    argv.insert(1, '-B')
    return argv, '/evaluator/cases/'+case_id, 'acceptance.json'


def namespace_command(cfg, paths, command, cwd, uid):
    argv = ['/usr/bin/bwrap', '--unshare-pid', '--unshare-net', '--unshare-ipc', '--unshare-uts',
            '--unshare-cgroup', '--die-with-parent', '--new-session', '--ro-bind', cfg['rootfs'], '/',
            '--proc', '/proc', '--dev', '/dev', '--perms', '1777', '--tmpfs', '/tmp',
            '--perms', '1777', '--tmpfs', '/dev/shm', '--tmpfs', '/home', '--dir', '/home/evaluator']
    for key, target in [('workspace','/work'),('source','/source'),('original','/original'),('tools','/tools'),('evaluator','/evaluator')]:
        argv += ['--ro-bind', str(paths[key]), target]
    argv += ['--ro-bind',str(ROOT/'controller/evaluation_driver.py'),'/evaluation_driver.py']
    argv += ['--bind', str(paths['evidence']), '/evidence']
    for path in cfg['system_mounts']:
        require(path in ('/usr','/lib','/lib64','/etc/ld.so.cache','/etc/ssl/certs','/etc/fonts','/etc/vulkan','/etc/alternatives'), 'Unqualified system path')
        argv += ['--ro-bind', path, path]
    common = read(Path(cfg['tools'])/'environment.json')
    env = {'PATH':'/tools/main-venv/bin:/usr/bin:/bin', 'HOME':'/home/evaluator', 'LANG':'C.UTF-8',
           'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONNOUSERSITE':'1',
           'PYTHONDONTWRITEBYTECODE':'1','UV_NO_CACHE':'1','WU_OVPHYSX_AUTO_PROVISION':'0',
           'WU_OVPHYSX_VENV_DIR':'/tools/ovphysx-venv'}
    if common.get('LD_LIBRARY_PATH'): env['LD_LIBRARY_PATH'] = common['LD_LIBRARY_PATH']
    argv += ['--clearenv', '--hostname', 'evaluator', '--chdir', cwd]
    for key,value in sorted(env.items()): argv += ['--setenv', key, value]
    argv += ['--', '/usr/bin/setpriv', '--reuid',str(uid),'--regid',str(uid),'--clear-groups',
             '--bounding-set=-all','--inh-caps=-all','--ambient-caps=-all','--no-new-privs','--', *command]
    return argv


def execute(cfg, request):
    require(os.geteuid() == 0, 'Trusted root evaluator supervisor required')
    run_id = identifier(request['run_id'])
    lane = request['lane']
    group = Path(cfg['lanes'][lane['id']]['cgroup'])
    require(populated(group) == 0, 'Author/evaluator lane occupied')
    limits = check_limits(group, lane)
    freeze = read(request['freeze_file'])
    require(freeze.get('frozen') is True and freeze.get('status') == 'PASS', 'Evaluator must be frozen')
    for row in freeze['files']:
        if row['path'].startswith('evaluator/'):
            require(sha256(ROOT/row['path']) == row['sha256'], 'Frozen evaluator file changed: '+row['path'])
    out = Path(cfg['runs_root'])/run_id
    out.mkdir(mode=0o755)
    for name in ('workspace','home','private'): (out/name).mkdir(mode=0o700)
    uid = 21000
    volumes = mount_run_storage(out, uid, {'workspace_bytes':8*2**30,'home_bytes':8*2**30})
    (out/'home').chmod(0o755)
    evidence = out/'home/evidence'; evidence.mkdir(mode=0o700); os.chown(evidence,uid,uid)
    write_new(out/'private/storage.json', {'volumes':volumes,'scope':'Post-author independent evaluation'})
    def enter():
        (group/'cgroup.procs').write_text(str(os.getpid()))
        os.sched_setaffinity(0,lane['cpu_affinity'])
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    restore = [sys.executable,'-B',str(Path(__file__).resolve()),'--restore',request['workspace_archive'],
               '--expected-sha',request['workspace_archive_sha256'],'--destination',str(out/'workspace')]
    subprocess.run(restore, preexec_fn=enter, check=True, timeout=900, env={'PATH':'/usr/bin:/bin'})
    preliminary = inspect_declaration(out/'workspace', request['declaration'], request['case_id'])
    if preliminary:
        write_new(evidence/'submission_rejection.json', preliminary)
        write_new(out/'private/evaluation.json', preliminary | {'run_id':run_id,'cgroup_populated':populated(group)})
        return preliminary
    command,cwd,report = evaluation_command(request['case_id'], request['declaration'],read(request['invocations_file']))
    # The wrapper validates the submitted USD without executing its scripts;
    # concrete parser rejection is distinct from unavailable evaluator machinery.
    command=[command[0],'-B','/evaluation_driver.py','--usd','/work/'+request['declaration']['final_scene'],
             '--output','/evidence/parser_preflight.json','--',*command]
    paths={'workspace':out/'workspace','source':request['source'],'original':request['original'],
           'tools':cfg['tools'],'evaluator':ROOT/'evaluator','evidence':evidence}
    argv=namespace_command(cfg,paths,command,cwd,uid)
    write_new(out/'private/launch.json', {'run_id':run_id,'case_id':request['case_id'],'command':command,
              'workspace_archive_sha256':request['workspace_archive_sha256'],'limits':limits,
              'network':'private unconnected namespace','author_workspace':'read-only','auth':'none',
              'freeze_sha256':sha256(request['freeze_file']),'supervisor_sha256':sha256(__file__)})
    start=time.monotonic(); timed_out=False
    try:
        with (out/'private/stdout.log').open('x') as stdout, (out/'private/stderr.log').open('x') as stderr:
            process=subprocess.Popen(argv,stdout=stdout,stderr=stderr,preexec_fn=enter,env={'PATH':'/usr/bin:/bin'})
            try: process.wait(timeout=9000)
            except subprocess.TimeoutExpired: timed_out=True; kill_and_reap(group); process.wait(timeout=15)
    finally: kill_and_reap(group)
    path=evidence/'result'/report
    raw=read(path) if path.is_file() else {}
    status=raw.get('status')
    status={'accepted':'PASS','not_accepted':'FAIL','inconclusive':'INCONCLUSIVE'}.get(status,status)
    if status not in ('PASS','FAIL','INCONCLUSIVE'): status='INCONCLUSIVE'
    preflight=read(evidence/'parser_preflight.json') if (evidence/'parser_preflight.json').is_file() else {}
    if preflight.get('status')=='FAIL': status='FAIL'
    receipt={'run_id':run_id,'case_id':request['case_id'],'status':status,'report':str(path.relative_to(out)),
             'report_sha256':sha256(path) if path.is_file() else None,'elapsed_seconds':time.monotonic()-start,
             'returncode':process.returncode,'timed_out':timed_out,'cgroup_populated':populated(group)}
    write_new(out/'private/evaluation.json',receipt)
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--restore'); parser.add_argument('--expected-sha'); parser.add_argument('--destination')
    parser.add_argument('--config'); parser.add_argument('--request')
    args=parser.parse_args()
    if args.restore: restore_workspace(args.restore,args.expected_sha,args.destination)
    else:
        check_host_owned(args.config); check_host_owned(args.request)
        print(json.dumps(execute(read(args.config),read(args.request))))
