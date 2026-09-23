"""Descriptive paired comparison; never infer missing verdicts, usage or labor."""
import argparse
import json
import math
from pathlib import Path

from accounting import per_accepted, reference_run

ARMS = ('plain_astra', 'content_agents')
STATES = ('PASS', 'FAIL', 'INCONCLUSIVE')


def summarize(rows, rates):
    assert all(x['independent_status'] in STATES for x in rows)
    accepted = sum(x['independent_status'] == 'PASS' for x in rows)
    positive = [x for x in rows if x.get('claimed_accepted') is True]
    resolved_positive = [x for x in positive if x['independent_status'] != 'INCONCLUSIVE']
    false_passes = sum(x['independent_status'] == 'FAIL' for x in positive)
    costs = [reference_run(x['usage_ledger'], rates) for x in rows]
    lower = sum(x['standard_api_equivalent_usd_lower'] for x in costs)
    upper = (sum(x['standard_api_equivalent_usd_upper'] for x in costs)
             if all(x['standard_api_equivalent_usd_upper'] is not None for x in costs) else None)
    def observed_sum(key):
        values = [x.get(key) for x in rows]
        assert all(v is None or (isinstance(v, (float,int)) and not isinstance(v,bool) and math.isfinite(v) and v >= 0) for v in values)
        return sum(values) if values and all(v is not None for v in values) else None
    elapsed = observed_sum('author_elapsed_seconds')
    lane_seconds = observed_sum('allocated_lane_seconds')
    review_minutes = observed_sum('human_review_minutes')
    return {'attempts':len(rows), 'accepted':accepted,
        'verdict_counts':{k:sum(x['independent_status']==k for x in rows) for k in STATES},
        'acceptance_fraction':accepted/len(rows) if rows else None,
        'false_passes':false_passes, 'resolved_author_positive_count':len(resolved_positive),
        'false_pass_fraction_of_resolved_author_positives':false_passes/len(resolved_positive) if resolved_positive else None,
        'unresolved_author_positives':len(positive)-len(resolved_positive),
        'missing_author_claims':sum(x.get('claimed_accepted') is None for x in rows),
        'author_elapsed_seconds':elapsed, 'author_seconds_per_accepted':per_accepted(elapsed,accepted),
        'allocated_gpu_lane_hours':lane_seconds/3600 if lane_seconds is not None else None,
        'gpu_lane_hours_per_accepted':per_accepted(lane_seconds/3600 if lane_seconds is not None else None,accepted),
        'standard_api_equivalent_usd_lower':lower, 'standard_api_equivalent_usd_upper':upper,
        'api_equivalent_usd_per_accepted_lower':per_accepted(lower,accepted),
        'api_equivalent_usd_per_accepted_upper':per_accepted(upper,accepted),
        'cost_complete_runs':sum(x['usage_complete'] for x in costs),
        'human_interventions':sum(len(x['human_interventions']) for x in rows),
        'human_review_minutes':review_minutes, 'human_review_minutes_per_accepted':per_accepted(review_minutes,accepted),
        'invoice_usd':None, 'horde_usd':None, 'human_usd':None,
        'cost_scope':'All attempts in this reported subset, including rejected and inconclusive assets. Public API reference only; incomplete terminal usage has unknown upper cost.'}


def compare(rows, rates):
    assert len(rows) == 20 and len({x['run_id'] for x in rows}) == 20
    cases = sorted({x['case_id'] for x in rows});assert len(cases)==10
    pairs = []; selected = []
    for case in cases:
        members = {x['arm']:x for x in rows if x['case_id']==case}
        assert set(members)==set(ARMS)
        assert all(type(x['protocol_eligible']) is bool for x in members.values())
        eligible = all(x['protocol_eligible'] for x in members.values())
        if eligible:
            for field in ('lane_id','task_sha256','input_sha256','source_sha256','protocol_sha256'):
                assert len({x[field] for x in members.values()})==1
            selected.extend(members.values())
        pairs.append({'case_id':case,'eligible':eligible,
                      **{arm:{'run_id':members[arm]['run_id'],'status':members[arm]['independent_status'],
                              'claimed_accepted':members[arm].get('claimed_accepted')} for arm in ARMS}})
    resolved = [p for p in pairs if p['eligible'] and all(p[a]['status'] in ('PASS','FAIL') for a in ARMS)]
    wins = sum(p['content_agents']['status']=='PASS' and p['plain_astra']['status']=='FAIL' for p in resolved)
    losses = sum(p['plain_astra']['status']=='PASS' and p['content_agents']['status']=='FAIL' for p in resolved)
    discordant=wins+losses
    pvalue=min(1.0,2*sum(math.comb(discordant,i) for i in range(min(wins,losses)+1))/2**discordant) if discordant else None
    n=sum(p['eligible'] for p in pairs)
    arms={a:summarize([x for x in selected if x['arm']==a],rates) for a in ARMS}
    return {'schema_version':'paired-comparison.v2','eligible_pairs':n,'pairs':pairs,
            'eligible_pair_arms':arms,'all_attempts_by_arm':{a:summarize([x for x in rows if x['arm']==a],rates) for a in ARMS},
            'acceptance_difference_content_minus_plain':(arms['content_agents']['accepted']-arms['plain_astra']['accepted'])/n if n else None,
            'resolved_pairs':len(resolved),'content_only_pass':wins,'plain_only_pass':losses,
            'exploratory_exact_sign_p_two_sided':pvalue,
            'inference_limit':'Ten previously used public development assets, one authored output per arm. Five physical seeds are not independent author replicates. No claim of population benefit or human labor savings.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rows',type=Path,required=True)
    p.add_argument('--rates',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=compare(json.loads(a.rows.read_text())['runs'],json.loads(a.rates.read_text()))
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
