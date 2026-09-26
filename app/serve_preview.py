"""Serve the built preview the way a static host has to serve it.

Two audiences: someone who wants to see the preview without standing up any web
server, and someone wiring their own edge who wants the deployment requirements
written as running code rather than prose. It uses only the standard library, so it
needs neither the project's Python environment nor Caddy/nginx/Docker.

    python -m app.serve_preview                # http://127.0.0.1:8741/
    python -m app.serve_preview --port 9000

What it enforces, and why each one matters (docs/PREVIEW.md §4 and §8):

- `/` serves the preview page. In a deployment the file is published as the site's
  index; here it is served by name, so `/` behaves the same either way.
- `.wasm` as application/wasm, `.md` as text/plain; charset=utf-8. The charset is not
  cosmetic: the report is Chinese and without it browsers may fall back to a local
  default (GBK) and show mojibake.
- Byte ranges. The package holds two ~97.6 MiB files; without ranges a dropped
  connection means starting over.
- Content-addressed files get `immutable`; the manifest and HTML do not, so a new
  package is picked up on the next load.
- No directory listing, and no path escaping the served roots.
- Bytes are sent verbatim, which is what the page's own sha256 check depends on.
"""
import argparse
import http.server
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
# The page reads the package from this prefix in dev, here, and in production. Keeping
# it identical is what lets <meta name="preview-package"> stay untouched everywhere
# except a CDN deployment.
PACKAGE_PREFIX = '/artifacts/preview/'
STATIC_ROOT = ROOT / 'web' / 'dist'
PACKAGE_ROOT = ROOT / 'artifacts' / 'preview'
ENTRY = 'preview.html'

MIME = {'.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
        '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
        '.json': 'application/json; charset=utf-8', '.wasm': 'application/wasm',
        '.vrm': 'model/gltf-binary', '.png': 'image/png', '.md': 'text/plain; charset=utf-8',
        '.xml': 'application/xml; charset=utf-8', '.bin': 'application/octet-stream'}
RANGE = re.compile(r'^bytes=(\d*)-(\d*)$')


# These carry a stable name but their content changes when the package is
# re-exported, so they must revalidate. Only content-addressed files can be cached
# hard: caching body_config.json for a year would pair a fresh manifest and fresh
# tensors with a stale physics interface.
FIXED_NAMES = {'meta.json', 'body_config.json', 'scene.xml', 'yumi.xml'}


def cache_control(relative, in_package):
    if in_package and Path(relative).name in FIXED_NAMES:
        return 'no-cache'
    if in_package or relative.startswith('assets/'):
        return 'public, max-age=31536000, immutable'
    if relative.endswith('.html'):
        return 'no-cache'
    if relative.endswith('.vrm'):
        return 'public, max-age=604800'
    return None


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'preview-static'

    def do_GET(self):
        self.respond(body=True)

    def do_HEAD(self):
        self.respond(body=False)

    def resolve(self):
        """Map the request path onto (file, relative-path, served-from-package)."""
        path = self.path.split('?')[0].split('#')[0]
        if path == '/':
            path = '/' + ENTRY
        path = unquote(path)
        if path.startswith(PACKAGE_PREFIX):
            root, relative, in_package = PACKAGE_ROOT, path[len(PACKAGE_PREFIX):], True
        elif path.startswith('/'):
            root, relative, in_package = STATIC_ROOT, path[1:], False
        else:
            return None
        candidate = os.path.normpath(os.path.join(root, relative))
        # Containment: ".." is collapsed above, so anything that climbed out of its
        # root fails this test instead of reading a neighbouring file.
        if not candidate.startswith(str(root) + os.sep):
            return None
        return Path(candidate), relative, in_package

    def respond(self, body):
        resolved = self.resolve()
        if resolved is None:
            return self.fail(404, 'Not found')
        file, relative, in_package = resolved
        if not file.is_file():
            return self.fail(404, 'Not found')
        size = file.stat().st_size
        content_type = MIME.get(file.suffix.lower(), 'application/octet-stream')

        start, end = 0, size - 1
        ranged = False
        header = self.headers.get('Range')
        if header:
            match = RANGE.match(header.strip())
            if not match:
                return self.fail(416, 'Malformed range')
            first, last = match.group(1), match.group(2)
            if first == '' and last == '':
                return self.fail(416, 'Empty range')
            if first == '':
                start, end = max(0, size - int(last)), size - 1
            else:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            if start > end or start >= size:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            ranged = True

        self.send_response(206 if ranged else 200)
        self.send_header('Content-Type', content_type)
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end - start + 1))
        if ranged:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        control = cache_control(relative, in_package)
        if control:
            self.send_header('Cache-Control', control)
        self.end_headers()
        if not body:
            return
        with file.open('rb') as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = stream.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def fail(self, code, message):
        payload = message.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        if os.environ.get('PREVIEW_QUIET') != '1':
            sys.stderr.write(f'{self.address_string()} {fmt % args}\n')


class Server(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8741)
    args = parser.parse_args()

    if not (STATIC_ROOT / ENTRY).is_file():
        raise SystemExit(f'{STATIC_ROOT / ENTRY} is missing — run `npm run build` first')
    if not (PACKAGE_ROOT / 'meta.json').is_file():
        print(f'warning: {PACKAGE_ROOT}/meta.json is missing; the page will fail to load '
              f'weights — run `python -m app.export_preview_weights`', file=sys.stderr)

    print(f'preview:  http://{args.host}:{args.port}/')
    print(f'  static   {STATIC_ROOT}')
    print(f'  package  {PACKAGE_ROOT} at {PACKAGE_PREFIX}')
    Server((args.host, args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
