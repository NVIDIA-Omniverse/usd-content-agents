"""Strict metadata-only admission for the pinned Codex client.

Client UUIDs associate requests with independently retained own-thread turns;
they do not attest UI effort or authenticate arbitrary caller-generated records.
"""
import json
import uuid

from common import object_hash, require

REQUIRED = ('session_id', 'thread_id', 'turn_id')
OPTIONAL = ('parent_thread_id', 'parent_turn_id', 'root_turn_id')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def strict_json(value):
    return json.loads(value, object_pairs_hook=unique_object)


def canonical_uuid(value):
    require(isinstance(value, str) and len(value) == 36, 'Expected canonical UUID')
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        raise ValueError('Malformed UUID') from None
    require(str(parsed) == value, 'Expected canonical UUID')
    return value


def project(body):
    metadata = body.get('client_metadata')
    require(isinstance(metadata, dict) and len(metadata) <= 256, 'Missing/bounded client metadata required')
    serialized = metadata.get('x-codex-turn-metadata')
    require(isinstance(serialized, str) and 0 < len(serialized.encode()) <= 1024 * 1024,
            'Missing/bounded canonical turn metadata required')
    canonical = strict_json(serialized)
    require(isinstance(canonical, dict), 'Turn metadata must be an object')
    result = {}
    for key in REQUIRED:
        a = canonical_uuid(canonical.get(key))
        b = canonical_uuid(metadata.get(key))
        require(a == b, 'Canonical/flat request identity conflict')
        result[key] = a
    for key in OPTIONAL:
        flat_key = 'x-codex-parent-thread-id' if key == 'parent_thread_id' else key
        values = [canonical_uuid(source[name]) for source, name in
                  ((canonical, key), (metadata, flat_key)) if name in source]
        require(len(set(values)) <= 1, 'Canonical/flat optional identity conflict')
        if values:
            result[key] = values[0]
    return dict(result, identity_sha256=object_hash(result))


def validate_projection(value):
    require(isinstance(value, dict), 'Request admission identity missing')
    require(set(REQUIRED) <= set(value) <= set(REQUIRED + OPTIONAL + ('identity_sha256',)),
            'Unexpected admission identity fields')
    identity = {key: canonical_uuid(val) for key, val in value.items() if key != 'identity_sha256'}
    require(value.get('identity_sha256') == object_hash(identity), 'Admission identity digest mismatch')
    return identity
