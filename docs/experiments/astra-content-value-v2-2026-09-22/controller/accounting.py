"""API-equivalent reference accounting; never infer invoice or missing usage."""
import json
from pathlib import Path


def reference_request(request, rates):
    usage = request.get('usage')
    if usage is None:
        return {'observed': False, 'standard_usd_lower': 0.0,
                'standard_usd_upper': None, 'reason': 'Terminal usage unavailable'}
    n, out = usage['input_tokens'], usage['output_tokens']
    assert type(n) is int and type(out) is int and min(n, out) >= 0
    assert usage['total_tokens'] == n + out
    details = usage.get('input_tokens_details') or {}
    cached = details.get('cached_tokens')
    writes = details.get('cache_write_tokens')
    for value in (cached, writes):
        assert value is None or (type(value) is int and 0 <= value <= n)
    assert cached is None or writes is None or cached + writes <= n
    long_context = n > rates['long_context_input_threshold_exclusive']
    r = rates['long_context_usd_per_million' if long_context else 'standard_usd_per_million']
    # The bounds cover every partition consistent with available usage. Cache
    # reads/writes are input subsets; reasoning tokens are an output subset.
    if cached is None:
        low_cached, high_cached = n - (writes or 0), 0
    else:
        low_cached = high_cached = cached
    low_writes = writes or 0
    high_writes = writes if writes is not None else n - high_cached
    def price(c, w):
        return ((n-c-w)*r['uncached_input'] + c*r['cached_input'] +
                w*r['cache_write'] + out*r['output']) / 1_000_000
    lower, upper = price(low_cached, low_writes), price(high_cached, high_writes)
    assert 0 <= lower <= upper
    tier = request.get('response_tier')
    multipliers = {'default': 1, 'priority': 2, 'flex': .5, 'batch': .5}
    multiplier = multipliers.get(tier)
    return {'observed': True, 'long_context': long_context,
            'standard_usd_lower': lower, 'standard_usd_upper': upper,
            'cached_input_known': cached is not None, 'cache_write_known': writes is not None,
            'response_tier': tier, 'known_tier_multiplier': multiplier,
            'returned_tier_reference_usd_lower': lower*multiplier if multiplier else None,
            'returned_tier_reference_usd_upper': upper*multiplier if multiplier else None}


def reference_run(usage_ledger, rates):
    rows = [reference_request(x, rates) for x in usage_ledger['requests']]
    observed = sum(x['observed'] for x in rows)
    complete = bool(rows) and observed == len(rows) and usage_ledger.get('complete') is True
    return {'run_id': usage_ledger['run_id'], 'request_count': len(rows),
            'requests_with_terminal_usage': observed, 'usage_complete': complete,
            'standard_api_equivalent_usd_lower': sum(x['standard_usd_lower'] for x in rows),
            'standard_api_equivalent_usd_upper': sum(x['standard_usd_upper'] for x in rows) if complete else None,
            'invoice_usd': None, 'horde_usd': None, 'human_usd': None,
            'scope': 'Normalized standard-tier public API reference, with unknown cache-write partitions bounded. This is not a ChatGPT invoice. Missing usage is not zero.',
            'requests': rows}


def per_accepted(total, accepted):
    assert type(accepted) is int and accepted >= 0
    return total / accepted if total is not None and accepted else None


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ledger', type=Path, required=True)
    p.add_argument('--rates', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = reference_run(json.loads(a.ledger.read_text()), json.loads(a.rates.read_text()))
    with a.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
