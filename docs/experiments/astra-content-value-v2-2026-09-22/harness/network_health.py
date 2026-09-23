#!/usr/bin/env python3
"""Read-only transport gate; no credential, prompt, or model request is used."""
import argparse
import http.client
import json
import os
import subprocess
import time
from pathlib import Path

from common import read, require, sha256, write_new


def probe(port=18862, timeout=5):
    result = {}
    for method in ('GET', 'POST'):
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
        try:
            # An invalid token must be rejected before model/ledger admission.
            headers = {'Authorization': 'Bearer synthetic-health-invalid-token', 'Content-Type': 'application/json'}
            conn.request(method, '/v1/responses', body=b'{}' if method == 'POST' else None, headers=headers)
            response = conn.getresponse()
            result[method] = response.status
            response.read(4096)
        except OSError as exc:
            result[method] = {'transport_error': type(exc).__name__}
        finally:
            conn.close()
    return result


def check_status(result):
    require(result == {'GET': 405, 'POST': 403}, 'Gateway unavailable or rejected route contract changed')


def verify(namespace, bridge_receipt, port=18862):
    require(os.geteuid() == 0, 'Trusted network supervisor required')
    record = read(bridge_receipt)
    pid = record['pid']; require(type(pid) is int and pid > 1, 'Invalid bridge PID')
    process = Path('/proc') / str(pid)
    require(process.exists(), 'Bridge process has exited')
    command = (process / 'cmdline').read_bytes().split(b'\0')
    script = Path(__file__).with_name('network_bridge.py').resolve()
    require(os.fsencode(str(script)) in command, 'Bridge PID has been reused')
    require(os.fsencode(str(namespace)) in command, 'Bridge belongs to another namespace')
    cgroup = (process / 'cgroup').read_text().strip()
    require(cgroup.endswith('/astra-v2-supervisor'), 'Bridge must remain outside author kill cgroups')
    require(f'net:[{os.stat(namespace).st_ino}]' == record['private_namespace'], 'Namespace identity changed')
    host = probe(port); check_status(host)
    completed = subprocess.run(['/usr/bin/nsenter', '--net=' + str(namespace), '/usr/bin/python3', '-B',
                                str(Path(__file__).resolve()), '--probe', '--port', str(port)],
                               capture_output=True, text=True, timeout=15, check=True)
    private = json.loads(completed.stdout); check_status(private)
    return {'schema_version': 'model-transport-health.v2', 'checked_unix': time.time(),
            'host_status': host, 'private_status': private, 'bridge_pid': pid,
            'bridge_cgroup': cgroup, 'bridge_receipt_sha256': sha256(bridge_receipt),
            'bridge_code_sha256': sha256(script), 'health_code_sha256': sha256(__file__),
            'private_namespace': record['private_namespace'], 'passed': True,
            'model_calls': 0, 'scope': 'Live fixed gateway transport; not model/native compatibility proof'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--probe', action='store_true')
    parser.add_argument('--port', type=int, default=18862); parser.add_argument('--namespace')
    parser.add_argument('--bridge-receipt'); parser.add_argument('--output')
    args = parser.parse_args()
    result = probe(args.port) if args.probe else verify(args.namespace, args.bridge_receipt, args.port)
    if args.output: write_new(args.output, result)
    print(json.dumps(result))
