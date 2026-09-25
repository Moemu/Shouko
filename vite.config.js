// Two HTML entries under one root: the studio workbench (index.html) and the
// serverless preview (preview.html). Both import three, so Rollup splits it into
// a shared chunk instead of shipping a copy per page.
//
// The preview additionally reads two directories that sit outside the vite root:
// the generated weight package and the pinned MuJoCo-WASM build. Production serves
// those from nginx and they are deliberately not build inputs — the package is a
// ~200MB publishing artifact, not build output — but the dev and preview servers
// have to reach them, which is what serveExternal covers.
import { defineConfig } from 'vite';
import { createReadStream, readFileSync, statSync } from 'node:fs';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const PROJECT_ROOT = fileURLToPath(new URL('.', import.meta.url));

const EXTERNAL = {
  '/artifacts/preview/': resolve(PROJECT_ROOT, 'artifacts/preview'),
  '/bench_mujoco/public/': resolve(PROJECT_ROOT, 'bench_mujoco/public'),
};

const MIME = {
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.json': 'application/json; charset=utf-8',
  '.xml': 'application/xml; charset=utf-8',
  '.vrm': 'model/gltf-binary',
  '.bin': 'application/octet-stream',
};

/** Expand `<!-- @include views/x.html -->` and drop blocks that need a server.
 *
 * The results and research views are shared by both pages, so their markup has one
 * source instead of a hand-copied fork. The static preview declares
 * <meta name="studio-server" content="none">, and the blocks indexed @server-only
 * (the training console, the evaluation controls) are removed from it outright
 * rather than hidden at runtime — a page with no backend has no use for the markup,
 * and hiding it would still ship it and leave it re-enableable.
 *
 * Fragments carry their own indentation, so the include line's indent is ignored. */
function viewFragments() {
  // The whole-line alternatives come first so a marker sitting alone on a line takes
  // its newline with it; a marker inside a line (the evaluation heading) matches on
  // its own. Dropping the markers unconditionally is what lets the server page's
  // expansion be compared byte-for-byte against the markup before the split.
  // The whole-line alternatives require the line terminator, so they only match a
  // marker that ends its line. A marker followed by more markup falls through to the
  // inline alternative, which leaves the line's own indentation intact.
  const TOKENS = /(^[ \t]*<!-- @server-only -->[ \t]*\r?\n|^[ \t]*<!-- @end -->[ \t]*\r?\n|<!-- @server-only -->|<!-- @end -->)/m;
  return {
    name: 'preview-view-fragments',
    enforce: 'pre',
    transformIndexHtml: {
      order: 'pre',
      handler: (html) => {
        const expanded = html.replace(
          /^[ \t]*<!-- @include ([^\s]+) -->[ \t]*$/gm,
          // Strip the fragment's trailing newline including a CR: the include line
          // keeps its own terminator, so leaving the CR behind yields "\r\r\n".
          (_, name) => readFileSync(resolve(PROJECT_ROOT, 'web', name), 'utf8').replace(/\r?\n$/, ''));
        const keepWrapped = !/name="studio-server"[^>]*content="none"/.test(expanded);
        let skip = false, out = '';
        for (const part of expanded.split(TOKENS)) {
          if (part === undefined) continue;
          if (part.includes('<!-- @server-only -->')) { skip = !keepWrapped; continue; }
          if (part.includes('<!-- @end -->')) { skip = false; continue; }
          if (!skip) out += part;
        }
        // A page with no backend must not ship links into an API it cannot reach. The
        // shared views link the report and the raw evidence JSON; the page refills
        // both from what it actually has (the evidence name is content-addressed, so
        // only the runtime can know it).
        return keepWrapped ? out : out.replace(/href="\/api\/[^"]*"/g, 'href="#"');
      },
    },
  };
}

/** Serve files from outside the vite root so the preview works in dev.
 *
 * Production has nginx in this role, so this only ever runs for `vite` and
 * `vite preview` — it is not a build input. */
function serveExternal() {
  const mount = (middlewares) => {
    for (const [prefix, directory] of Object.entries(EXTERNAL)) {
      middlewares.use((request, response, next) => {
        const path = (request.url || '').split('?')[0];
        if (!path.startsWith(prefix)) return next();
        // normalize() collapses "..", and the containment test then rejects
        // anything that climbed out of the mounted directory.
        const file = join(directory, normalize(decodeURIComponent(path.slice(prefix.length))));
        if (!file.startsWith(directory + sep)) return next();
        let stats;
        try {
          stats = statSync(file);
        } catch {
          return next();
        }
        if (!stats.isFile()) return next();
        response.setHeader('Content-Type', MIME[extname(file)] || 'application/octet-stream');
        response.setHeader('Content-Length', stats.size);
        createReadStream(file).pipe(response);
      });
    }
  };
  return {
    name: 'preview-external-assets',
    configureServer: (server) => mount(server.middlewares),
    configurePreviewServer: (server) => mount(server.middlewares),
  };
}

export default defineConfig({
  root: 'web',
  build: {
    outDir: 'dist',
    rollupOptions: {
      // Absolute, not root-relative: rollup resolves these itself, not against `root`.
      input: {
        studio: resolve(PROJECT_ROOT, 'web/index.html'),
        preview: resolve(PROJECT_ROOT, 'web/preview.html'),
      },
    },
  },
  plugins: [viewFragments(), serveExternal()],
});
