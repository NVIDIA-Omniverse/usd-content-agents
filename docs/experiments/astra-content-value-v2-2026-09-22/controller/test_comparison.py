import copy
import json
from pathlib import Path
import unittest
from compare_results import compare, summarize

RATES=json.loads((Path(__file__).parents[1]/'protocol/api_reference_rates.json').read_text())

def run(status='PASS',claimed=True,pending=False):
    return {'independent_status':status,'claimed_accepted':claimed,'author_elapsed_seconds':10,
            'allocated_lane_seconds':12,'human_review_minutes':None,'human_interventions':[],
            'usage_ledger':{'run_id':'synthetic','complete':not pending,'requests':[{
                'response_tier':'default','usage':None if pending else {'input_tokens':100,'output_tokens':10,'total_tokens':110,
                'input_tokens_details':{'cached_tokens':0,'cache_write_tokens':0}}}]}}

class ComparisonTests(unittest.TestCase):
    def test_false_pass_excludes_unresolved_claim(self):
        r=summarize([run('FAIL'),run('INCONCLUSIVE'),run('PASS')],RATES)
        self.assertEqual((r['false_passes'],r['resolved_author_positive_count'],r['unresolved_author_positives']),(1,2,1))
        self.assertEqual(r['false_pass_fraction_of_resolved_author_positives'],.5)

    def test_rejected_effort_included_and_unmeasured_labor_unknown(self):
        r=summarize([run('FAIL',False),run('PASS')],RATES)
        self.assertEqual(r['author_seconds_per_accepted'],20)
        self.assertIsNone(r['human_review_minutes'])
        self.assertEqual(r['human_interventions'],0)

    def test_zero_acceptance_and_missing_cost_never_zero_unit_cost(self):
        r=summarize([run('FAIL',False,True)],RATES)
        self.assertIsNone(r['api_equivalent_usd_per_accepted_lower'])
        self.assertIsNone(r['standard_api_equivalent_usd_upper'])
        self.assertIsNone(r['author_seconds_per_accepted'])

    def test_pair_exclusion_is_not_selected_by_acceptance(self):
        rows=[]
        for i in range(10):
            for arm in ('plain_astra','content_agents'):
                r=run('PASS' if arm=='plain_astra' else 'FAIL',True)
                r.update(run_id=f'{i}_{arm}',case_id=f'{i:02d}',arm=arm,protocol_eligible=i!=0,
                         lane_id='same',task_sha256='same',input_sha256='same',source_sha256='same',protocol_sha256='same')
                rows.append(r)
        result=compare(rows,RATES)
        self.assertEqual(result['eligible_pairs'],9)
        self.assertEqual(result['plain_only_pass'],9)
        self.assertEqual(result['acceptance_difference_content_minus_plain'],-1)
        self.assertEqual(result['all_attempts_by_arm']['plain_astra']['attempts'],10)

if __name__=='__main__':unittest.main()
