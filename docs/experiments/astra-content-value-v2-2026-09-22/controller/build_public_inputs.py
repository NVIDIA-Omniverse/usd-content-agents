"""Package identical author-visible inputs without copying hidden evaluator data.

Draft generation is allowed before qualification; final creation requires the
root's complete, hash-bound evaluator qualification receipt. Never mutates v1.
"""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'harness'))
from common import inventory, object_hash, read, require, sha256, write_new


def build(destination, qualification=None):
    require(not destination.exists(), 'Input snapshot is create-only')
    if qualification:
        q = read(qualification)
        require(q.get('status') == 'PASS' and q.get('qualified_cases') ==
                [x['case_id'] for x in read(ROOT/'protocol/dataset.json')['assets']],
                'Complete evaluator qualification required')
        for row in q['files']:
            require(sha256(ROOT/row['path']) == row['sha256'], 'Qualified evaluator input changed')
    result = []
    for asset in read(ROOT/'protocol/dataset.json')['assets']:
        case = asset['case_id']
        task_path = ROOT/('protocol/tasks/01_drawer.json' if case == '01_drawer'
                         else 'evaluator/tasks/'+case+'.json')
        task = read(task_path)
        task = {k:v for k,v in task.items() if not k.startswith('private_') and
                k not in ('freeze_basis', 'evaluator_qualification')}
        task['schema_version'] = 'astra-content-value-task.v2'
        task['frozen'] = qualification is not None
        task['source_root'] = '/source'
        task['primary_input'] = '/source/'+str(Path(asset['primary_input']).relative_to(asset['source_root']))
        task['mounted_inputs'] = 'Only this original source closure and public task inventory are mounted. Preserve original bytes; save all derivatives under /work.'
        if case == '01_drawer':
            manifest = read(ROOT/'provenance/source_manifests/01_drawer.json')
            source_inventory = {'case_id':case, 'source_primary_sha256':asset['primary_sha256'],
                'components':manifest['structure']['node_names'],
                'structure':manifest['structure'], 'units':manifest['units'],
                'source_geometry_modified':False,
                'role_mapping':{'drawer':'drawer_cabinet_drawer_01', 'cabinet':'drawer_cabinet',
                                'anchored_lower_drawers':['drawer_cabinet_drawer_02','drawer_cabinet_drawer_03','drawer_cabinet_drawer_04']}}
            # Correct prose spacing without altering any numeric acceptance data.
            for key in ('task','submission_contract'):
                task[key] = re.sub(r'\b(at least|free|of|common|to)(?=\d)', r'\1 ', task[key])
            write_new(destination/case/(case+'_source_inventory.json'), source_inventory)
        else:
            source_path = ROOT/'evaluator/tasks'/(case+'_source_inventory.json')
            expected = task.get('source_inventory_sha256', task.get('reference_inventory_sha256'))
            require(sha256(source_path) == expected, 'Public source inventory mismatch')
            target = destination/case/(case+'_source_inventory.json')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source_path.read_bytes())
        source_inventory_path = destination/case/(case+'_source_inventory.json')
        task['source_inventory_file'] = '/input/'+source_inventory_path.name
        task['source_inventory_sha256'] = sha256(source_inventory_path)
        if qualification:
            task['qualification_receipt_sha256'] = sha256(qualification)
        write_new(destination/case/'task.json',task)
        closure = inventory(destination/case)
        require(len(closure['files']) == 2, 'Only the task and its inventory are public')
        result.append({'case_id':case,'task_sha256':sha256(destination/case/'task.json'),
                       'input_sha256':object_hash(closure), 'files':closure['files'],
                       'source_root':asset['source_root'], 'source_manifest_sha256':asset['manifest_sha256']})
    write_new(destination/'inputs_manifest.json', {'frozen':qualification is not None,'cases':result})
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--qualification',type=Path)
    a=p.parse_args()
    result=build(a.output,a.qualification)
    print(json.dumps({'cases':len(result),'frozen':a.qualification is not None,'output':str(a.output)}))
