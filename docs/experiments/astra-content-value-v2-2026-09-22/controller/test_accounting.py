import json
from pathlib import Path
import unittest
from accounting import reference_request, reference_run, per_accepted

RATES = json.loads((Path(__file__).parents[1] / 'protocol/api_reference_rates.json').read_text())


def row(n=1000, out=100, details=None, tier='default'):
    return {'usage': {'input_tokens': n, 'output_tokens': out, 'total_tokens': n+out,
                      'input_tokens_details': details or {},
                      'output_tokens_details': {'reasoning_tokens': out}},
            'response_tier': tier}


class AccountingTests(unittest.TestCase):
    def test_subsets_are_not_double_counted(self):
        x = reference_request(row(details={'cached_tokens': 200, 'cache_write_tokens': 100}), RATES)
        self.assertAlmostEqual(x['standard_usd_lower'], .01345)
        self.assertEqual(x['standard_usd_lower'], x['standard_usd_upper'])

    def test_unknown_writes_are_bounded(self):
        x = reference_request(row(details={'cached_tokens': 200}), RATES)
        self.assertAlmostEqual(x['standard_usd_lower'], .0132)
        self.assertAlmostEqual(x['standard_usd_upper'], .0152)

    def test_unknown_cache_is_not_zero(self):
        x = reference_request(row(), RATES)
        self.assertAlmostEqual(x['standard_usd_lower'], .006)
        self.assertAlmostEqual(x['standard_usd_upper'], .0175)

    def test_long_context_applies_per_request(self):
        exact = {'cached_tokens': 0, 'cache_write_tokens': 0}
        a = reference_request(row(272000, 0, exact), RATES)
        b = reference_request(row(272001, 0, exact), RATES)
        self.assertFalse(a['long_context'])
        self.assertTrue(b['long_context'])
        self.assertAlmostEqual(a['standard_usd_lower'], 2.72)
        self.assertAlmostEqual(b['standard_usd_lower'], 5.44002)

    def test_unknown_tier_not_invented(self):
        x = reference_request(row(tier=None), RATES)
        self.assertIsNone(x['returned_tier_reference_usd_lower'])

    def test_priority_is_explicit(self):
        x = reference_request(row(tier='priority'), RATES)
        self.assertEqual(x['returned_tier_reference_usd_lower'], 2*x['standard_usd_lower'])

    def test_missing_terminal_cannot_make_complete_cost(self):
        ledger = {'run_id': 'test', 'complete': False, 'requests': [row(), {'usage': None}]}
        x = reference_run(ledger, RATES)
        self.assertFalse(x['usage_complete'])
        self.assertIsNone(x['standard_api_equivalent_usd_upper'])
        self.assertGreater(x['standard_api_equivalent_usd_lower'], 0)

    def test_zero_acceptances_has_no_finite_cost_per_acceptance(self):
        self.assertIsNone(per_accepted(100, 0))
        self.assertEqual(per_accepted(100, 4), 25)

    def test_conflicting_partitions_rejected(self):
        with self.assertRaises(AssertionError):
            reference_request(row(details={'cached_tokens': 900, 'cache_write_tokens': 200}), RATES)


if __name__ == '__main__':
    unittest.main()
