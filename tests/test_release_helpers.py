"""Dependency-free contracts for the public reproduction wrappers."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublicReleaseHelpersTest(unittest.TestCase):
    def test_external_seed_panels_and_fresh_tags(self):
        driver = load('reproduce_external')
        self.assertEqual(driver.PANELS['GSE148794']['seeds'], (42, 123, 2026, 7, 13))
        self.assertEqual(driver.PANELS['GSE305979']['seeds'], (42, 123, 2026))
        env = driver.run_environment('GSE149622', 7)
        self.assertEqual(env['LEWORLD_RESULT_TAG'], 'reproduction_ext_gse149622_v2')
        self.assertEqual(env['LEWORLD_TRAIN_TIMEPOINTS'], '0,1,2,3')
        self.assertEqual(env['LEWORLD_HOLDOUT_TIMEPOINTS'], '4')
        self.assertEqual(env['LEWORLD_EPOCHS'], '100')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')

    def test_tag_fragments_reject_traversal(self):
        helper = load('summarize_release_results')
        for value in ('..', '../escape', '/absolute', 'x/y'):
            with self.assertRaises(ValueError):
                helper.safe_fragment(value)
        self.assertEqual(helper.safe_fragment('reproduction_'), 'reproduction_')

    def test_historical_numeric_aggregates(self):
        helper = load('summarize_release_results')
        helper.verify_historical(helper.primary_summary('primary_internal_freeze'),
                                 helper.external_summary(''))

    def test_privacy_scan_detects_contact_without_echoing_it(self):
        helper = load('scan_release_privacy')
        contact = 'person' + '@' + 'invalid' + '.' + 'example'
        with tempfile.TemporaryDirectory(prefix='release_scan_test_') as folder:
            root = Path(folder)
            (root / 'example.txt').write_text(contact, encoding='utf-8')
            result = helper.scan(root)
            self.assertEqual(result['status'], 'FAIL')
            self.assertEqual(result['categories']['contact_address'], 1)
            self.assertNotIn(contact, json.dumps(result))

    def test_privacy_scan_does_not_open_denied_file(self):
        helper = load('scan_release_privacy')
        with tempfile.TemporaryDirectory(prefix='release_scan_test_') as folder:
            root = Path(folder)
            (root / 'auth.json').write_bytes(b'\xff\xfe\x00')
            result = helper.scan(root)
            self.assertEqual(result['categories']['blocked_path'], 1)
            self.assertEqual(result['text_files_scanned'], 0)

    def test_privacy_scan_inspects_decoded_json_paths(self):
        helper = load('scan_release_privacy')
        with tempfile.TemporaryDirectory(prefix='release_scan_test_') as folder:
            root = Path(folder)
            bad_path = '/' + 'home' + '/' + 'example'
            (root / 'example.json').write_text(json.dumps({'path': bad_path}), encoding='utf-8')
            result = helper.scan(root)
            self.assertEqual(result['categories']['machine_path'], 1)


if __name__ == '__main__':
    unittest.main()
