"""CPU/stdlib-only regression: python -m app.test_serve_preview."""
import hashlib
import json
import os
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app.serve_preview import Handler, PACKAGE_ROOT, STATIC_ROOT, Server

os.environ.setdefault('PREVIEW_QUIET', '1')

PACKAGE_META = PACKAGE_ROOT / 'meta.json'
PAGE = STATIC_ROOT / 'preview.html'


class ServePreviewTest(unittest.TestCase):
    """Exercises the served bytes over HTTP: this is the layer the JS package test
    cannot reach, and the layer a misconfigured edge gets wrong."""

    @classmethod
    def setUpClass(cls):
        cls.server = Server(('127.0.0.1', 0), Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f'http://127.0.0.1:{cls.server.server_address[1]}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def fetch(self, path, headers=None):
        """Returns (status, headers, body); HTTP errors come back instead of raising."""
        try:
            with urlopen(Request(self.base + path, headers=headers or {})) as response:
                return response.status, response.headers, response.read()
        except HTTPError as error:
            return error.code, error.headers, error.read()

    def package_files(self):
        return json.loads(PACKAGE_META.read_text())['files'] if PACKAGE_META.is_file() else {}

    @unittest.skipUnless(PAGE.is_file(), 'run `npm run build` first')
    def test_root_serves_the_preview_page_not_the_workbench(self):
        status, headers, body = self.fetch('/')
        self.assertEqual(status, 200)
        self.assertEqual(headers['Content-Type'], 'text/html; charset=utf-8')
        self.assertEqual(headers['Cache-Control'], 'no-cache')
        self.assertIn(b'preview-package', body)
        self.assertNotIn(b'trainButton', body, 'the workbench console must not be served')

    @unittest.skipUnless(PAGE.is_file(), 'run `npm run build` first')
    def test_content_addressed_assets_are_immutable(self):
        listing = sorted(p.name for p in (STATIC_ROOT / 'assets').glob('*.js'))
        self.assertTrue(listing, 'no built assets')
        status, headers, body = self.fetch(f'/assets/{listing[0]}')
        self.assertEqual(status, 200)
        self.assertEqual(headers['Cache-Control'], 'public, max-age=31536000, immutable')
        self.assertTrue(body)

    @unittest.skipUnless(PAGE.is_file(), 'run `npm run build` first')
    def test_report_keeps_its_charset(self):
        report = sorted((STATIC_ROOT / 'assets').glob('REPORT-*.md'))
        self.assertTrue(report, 'the report was not emitted')
        status, headers, _ = self.fetch(f'/assets/{report[0].name}')
        self.assertEqual(status, 200)
        self.assertEqual(headers['Content-Type'], 'text/plain; charset=utf-8')

    def test_directory_listing_is_refused(self):
        for path in ('/assets/', '/artifacts/preview/'):
            status, _, _ = self.fetch(path)
            self.assertEqual(status, 404, f'{path} should not be listed')

    def test_traversal_is_refused(self):
        for path in ('/%2e%2e/package.json',
                     '/artifacts/preview/%2e%2e%2f%2e%2e%2fpackage.json',
                     '/assets/%2e%2e%2f%2e%2e%2fpackage.json'):
            status, _, body = self.fetch(path)
            self.assertEqual(status, 404, path)
            self.assertNotIn(b'neuromechfly', body)

    @unittest.skipUnless(PACKAGE_META.is_file(), 'run the exporter first')
    def test_manifest_is_not_cached(self):
        status, headers, _ = self.fetch('/artifacts/preview/meta.json')
        self.assertEqual(status, 200)
        self.assertEqual(headers['Cache-Control'], 'no-cache')
        self.assertEqual(headers['Content-Type'], 'application/json; charset=utf-8')

    @unittest.skipUnless(PACKAGE_META.is_file(), 'run the exporter first')
    def test_wasm_and_binary_types(self):
        meta = json.loads(PACKAGE_META.read_text())
        # application/wasm is what WebAssembly.instantiateStreaming requires.
        status, headers, _ = self.fetch('/artifacts/preview/' + meta['mujoco']['wasm']['path'])
        self.assertEqual(status, 200)
        self.assertEqual(headers['Content-Type'], 'application/wasm')
        status, headers, _ = self.fetch('/artifacts/preview/' + meta['files']['values']['path'])
        self.assertEqual(status, 200)
        self.assertEqual(headers['Content-Type'], 'application/octet-stream')
        self.assertEqual(headers['Cache-Control'], 'public, max-age=31536000, immutable')
        self.assertEqual(headers['Accept-Ranges'], 'bytes')

    @unittest.skipUnless(PACKAGE_META.is_file(), 'run the exporter first')
    def test_range_requests_serve_exact_slices(self):
        name = self.package_files()['values']['path']
        path = '/artifacts/preview/' + name
        whole = (PACKAGE_ROOT / name).read_bytes()
        status, headers, body = self.fetch(path, {'Range': 'bytes=0-1023'})
        self.assertEqual(status, 206)
        self.assertEqual(headers['Content-Range'], f'bytes 0-1023/{len(whole)}')
        self.assertEqual(body, whole[:1024])
        status, headers, body = self.fetch(path, {'Range': f'bytes={len(whole) - 16}-'})
        self.assertEqual(status, 206)
        self.assertEqual(body, whole[-16:])
        status, headers, _ = self.fetch(path, {'Range': f'bytes={len(whole)}-'})
        self.assertEqual(status, 416)
        self.assertEqual(headers['Content-Range'], f'bytes */{len(whole)}')

    @unittest.skipUnless(PACKAGE_META.is_file(), 'run the exporter first')
    def test_served_bytes_match_the_manifest(self):
        """The page hashes what it receives, so this is the check that a misconfigured
        edge (rewriting, re-encoding) would break."""
        files = self.package_files()
        for key in ('ptr', 'bias', 'sample'):
            entry = files[key]
            status, _, body = self.fetch('/artifacts/preview/' + entry['path'])
            self.assertEqual(status, 200)
            self.assertEqual(hashlib.sha256(body).hexdigest(), entry['sha256'], f'{key} differs')


if __name__ == '__main__':
    unittest.main()
