"""Run with: python -m unittest discover -s tests -p test_checker_human_review.py"""
import importlib.util
import tempfile
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / 'scripts/checker_human_review.py'
SPEC = importlib.util.spec_from_file_location('review_audit', PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class ReviewAuditTests(unittest.TestCase):
    def reviews(self):
        return [dict(review_id='RV-1', assessor_id=a, bank_sha256='HASH',
                     explanation_verdict='Supported') for a in ['A', 'B']]

    def test_hash_mismatch_stops_scoring(self):
        with self.assertRaisesRegex(ValueError, 'VERSION MISMATCH'):
            audit.align_reviews(self.reviews(), [{}], 'OTHER', lambda _: 'RV-1', 1)

    def test_human_disagreement_requires_adjudication(self):
        rows = self.reviews()
        rows[1]['explanation_verdict'] = 'Incorrect'
        with self.assertRaisesRegex(ValueError, 'disagreement'):
            audit.align_reviews(rows, [{}], 'HASH', lambda _: 'RV-1', 1)

    def test_mismatched_identity_stops_scoring(self):
        with self.assertRaisesRegex(ValueError, 'Review ID'):
            audit.align_reviews(self.reviews(), [{}], 'HASH', lambda _: 'RV-2', 1)

    def test_one_assessor_is_not_two(self):
        with self.assertRaisesRegex(ValueError, 'two assessors'):
            audit.align_reviews(self.reviews()[:1], [{}], 'HASH', lambda _: 'RV-1', 1)

    def test_missing_items_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, 'every bank item'):
            audit.align_reviews(self.reviews(), [{'id': 'RV-1'}, {'id': 'RV-2'}],
                                'HASH', lambda r: r['id'], 2)

    def test_matching_reviews_are_accepted(self):
        result = audit.align_reviews(self.reviews(), [{}], 'HASH', lambda _: 'RV-1', 1)
        self.assertEqual(len(result), 1)

    def test_abstention_is_not_error_detection(self):
        rows = [{'human_verdict': 'Incorrect', 'checker_verdict': 'Insufficient evidence'},
                {'human_verdict': 'Supported', 'checker_verdict': 'Supported'}]
        result = audit.summarize(rows)
        self.assertEqual(result['explicit_error_detection_rate'], 0)
        self.assertEqual(result['abstention_rate'], .5)
        self.assertEqual(result['exact_verdict_agreement'], .5)

    def test_contradiction_takes_precedence(self):
        label, _ = audit.checker_verdict({'claims': [{'status': 'contradicted'}, {'status': 'unsupported'}]})
        self.assertEqual(label, 'Incorrect')

    def test_no_extracted_claims_is_not_supported(self):
        self.assertEqual(audit.checker_verdict({'claims': []})[0], 'Insufficient evidence')

    def test_unsupported_is_not_supported(self):
        self.assertEqual(audit.checker_verdict({'claims': [{'status': 'unsupported'}]})[0], 'Insufficient evidence')

    def test_all_extracted_claims_supported(self):
        self.assertEqual(audit.checker_verdict({'claims': [{'status': 'supported'}]})[0], 'Supported')

    def test_duplicate_review_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'reviews.csv'
            columns = ['bank_sha256','review_id','assessor_id','explanation_verdict',
                       'recommendation_verdict','claim_or_reason','reference_location','scope_clear']
            row = {'bank_sha256': 'HASH', 'review_id': 'RV-1', 'assessor_id': 'A',
                   'explanation_verdict': 'Supported','recommendation_verdict': 'Supported',
                   'claim_or_reason': 'Checked','reference_location': 'Table','scope_clear': 'Yes'}
            audit.write_csv(p, [row, row])
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                audit.read_reviews([p])


if __name__ == '__main__':
    unittest.main()
