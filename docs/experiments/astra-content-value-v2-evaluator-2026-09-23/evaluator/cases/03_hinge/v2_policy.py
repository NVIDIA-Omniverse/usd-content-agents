"""Public auxiliary-body policy, independent of any author's metadata claims."""


def auxiliary_checks(bindings, bodies, joints, source_map, contract):
    records = []
    controlled = set(contract.get('control', {}))
    if contract.get('control_default'):
        controlled.update(contract.get('required_joint_roles', []))
    mapped = {m['body_path'] for m in source_map}
    entries = {b['path']: b for b in bindings['bodies']}
    auxiliary = [p for p, b in bodies.items() if b['moving'] and p not in mapped]
    permitted = contract.get('allow_passive_auxiliary_bodies', False)
    records.append(('auxiliary_body_count', len(auxiliary) <= (8 if permitted else 0),
                    {'count': len(auxiliary), 'limit': 8 if permitted else 0}))
    required = set(contract['required_body_roles'])
    for path in auxiliary:
        incident = [j for j in joints.values() if path in (j['body0'], j['body1'])]
        neighbors = {j['body1'] if j['body0'] == path else j['body0'] for j in incident}
        declared = entries[path].get('auxiliary') == 'passive_constraint'
        passive = not any(j['role'] in controlled for j in incident)
        records.append(('passive_auxiliary_declared:' + path,
                        permitted and declared and bodies[path]['role'] not in required,
                        {'declaration': entries[path].get('auxiliary'), 'permitted_case': permitted}))
        records.append(('passive_auxiliary_connected:' + path, len(neighbors) >= 2,
                        {'distinct_neighbors': sorted(str(n) for n in neighbors)}))
        records.append(('passive_auxiliary_not_actuated:' + path, passive,
                        {'incident_joint_roles': [j['role'] for j in incident]}))
    # Every declared auxiliary must actually be auxiliary; labels cannot exempt
    # an intended original part or hide an actuated source part from its role.
    for path, entry in entries.items():
        if 'auxiliary' in entry:
            records.append(('auxiliary_label_is_source_free:' + path, path in auxiliary, None))
    return records
