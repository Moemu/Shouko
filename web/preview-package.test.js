// Checks the two generated artifacts the preview ships on, skipping either when it
// has not been produced. The package is checked against the loader's own file
// contract (FILE_KINDS); the built page is checked for asset URLs a deployed host
// could actually resolve.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { FILE_KINDS } from './brain-webgpu.js';

const root = new URL('../artifacts/preview/', import.meta.url);
if (!existsSync(new URL('meta.json', root))) {
  console.log('Preview package: none exported; skipped (run app/export_preview_weights.py).');
  process.exit(0);
}

const read = (path) => readFile(new URL(path, root));
const sha256 = (raw) => createHash('sha256').update(raw).digest('hex');
const meta = JSON.parse(await read('meta.json'));

for (const [key, entry] of Object.entries(meta.files)) {
  const spec = FILE_KINDS[key];
  assert.ok(spec, `meta.json declares a file kind the loader does not know: ${key}`);
  const raw = await read(entry.path);
  assert.equal(raw.length, spec.size(meta) * spec.type.BYTES_PER_ELEMENT,
    `${key} is ${raw.length} bytes, expected ${spec.size(meta)} elements`);
  assert.equal(sha256(raw), entry.sha256, `${key} does not match its recorded sha256`);
}

// The gather shader indexes neurons directly, so an out-of-range sample vertex
// reads unrelated memory and the panel would show plausible-looking nonsense.
const sampleRaw = await read(meta.files.sample.path);
const sample = new Int32Array(new Uint8Array(sampleRaw).buffer);
assert.equal(sample.length, meta.sample_count);
assert.ok(sample.every((index) => index >= 0 && index < meta.n), 'sample index out of range');
assert.ok(sample.every((index, i) => i === 0 || index > sample[i - 1]), 'sample indices not ascending');

const verifyHashed = async (label, entry) => {
  assert.ok(entry && entry.path && entry.sha256, `${label} is not recorded in meta.json`);
  assert.equal(sha256(await read(entry.path)), entry.sha256, `${label} (${entry.path}) doesn't match meta.json`);
};
for (const [key, entry] of Object.entries(meta.scene)) await verifyHashed(`scene.${key}`, entry);
for (const [key, entry] of Object.entries(meta.mujoco)) {
  if (entry && typeof entry === 'object') await verifyHashed(`mujoco.${key}`, entry);
}
assert.ok(meta.mujoco.version, 'the MuJoCo package version was not recorded');

// The exporter refuses to package evidence from another checkpoint and the page refuses
// to render it, so this pins the artifact's own claim in between the two. The studio
// applies the same binding in /api/evaluation.
const evidence = JSON.parse(await read(meta.evaluation.path));
await verifyHashed('evaluation', meta.evaluation);
assert.equal(evidence.checkpoint_sha256, meta.checkpoint_sha256, 'evidence belongs to another checkpoint');
assert.equal(evidence.checkpoint_sha256, meta.evaluation.checkpoint_sha256);
assert.equal(evidence.kind, 'held_out_locomotion');
assert.equal(evidence.complete, true, 'incomplete evidence must not ship');

const config = JSON.parse(await read('body_config.json'));
assert.equal(evidence.robot, config.robot, 'evidence was measured on another robot');
for (const [key, recorded] of Object.entries(meta.physics_interface)) {
  const local = key === 'default_angles' ? 'home' : key;
  const want = [recorded].flat(), got = [config[local]].flat();
  assert.equal(got.length, want.length, `${local} has ${got.length} entries, checkpoint wants ${want.length}`);
  want.forEach((value, i) => assert.ok(Math.abs(value - got[i]) < 1e-6, `${local}[${i}] drifted from the checkpoint`));
}

// Content-addressed names mean a second export cannot overwrite the first, so an
// undeclared file is either a stale revision or something that never got recorded.
const declared = new Set(['meta.json', 'body_config.json',
  ...Object.values(meta.files).map((entry) => entry.path),
  ...Object.values(meta.scene).map((entry) => entry.path),
  meta.mujoco.glue.path, meta.mujoco.wasm.path, meta.evaluation.path]);
const present = (await readdir(root)).filter((name) => name !== '.gitkeep');
assert.deepEqual(present.filter((name) => !declared.has(name)), [],
  'the package contains files meta.json does not declare');

console.log(`Preview package: ${Object.keys(meta.files).length} tensors + scene + `
  + `MuJoCo ${meta.mujoco.version} verified against checkpoint ${meta.checkpoint_sha256.slice(0, 8)} `
  + `(${meta.n} neurons, ${meta.sample_count} somata drawn).`);

// The preview was originally served straight from the repository root, so its
// asset URLs pointed at /web/... and /node_modules/... and no deployed host could
// resolve them. These assertions are what catch a return to that state.
const dist = new URL('../web/dist/', import.meta.url);
if (!existsSync(new URL('preview.html', dist))) {
  console.log('Built pages: web/dist not built; skipped (run npm run build).');
} else {
  const preview = await readFile(new URL('preview.html', dist), 'utf8');
  const studio = await readFile(new URL('index.html', dist), 'utf8');
  for (const [name, html] of [['preview.html', preview], ['index.html', studio]]) {
    for (const stale of ['/web/', '/node_modules/', 'importmap']) {
      assert.ok(!html.includes(stale), `${name} in web/dist still references ${stale}`);
    }
    assert.match(html, /src="\/assets\/[^"]+\.js"/, `${name} carries no bundled script entry`);
    for (const marker of ['@include', '@server-only', '<!-- @end -->']) {
      assert.ok(!html.includes(marker), `${name} still carries the build marker ${marker}`);
    }
    // Both pages share the results and research views from web/views.
    for (const id of ['evaluationRows', 'evaluationSummary', 'resultsCriteria', 'researchContext']) {
      assert.ok(html.includes(`id="${id}"`), `${name} is missing the shared view element ${id}`);
    }
  }
  // The static preview must not carry markup it has no backend for: the build removes
  // it outright rather than hiding it, so it cannot be re-enabled from the console.
  for (const id of ['trainButton', 'trainConfig', 'evaluateButton', 'evaluationProtocol', 'evaluationJob', 'hardwareText']) {
    assert.ok(!preview.includes(`id="${id}"`), `preview.html still ships the server-only element ${id}`);
    assert.ok(studio.includes(`id="${id}"`), `index.html lost the server-only element ${id}`);
  }
  // Beyond the shared invariants, the preview must stand alone on a static host:
  // every URL it names resolves to a file in dist/ or to the package the host
  // serves. index.html is excluded because it legitimately points at the studio
  // API, in-page anchors and cited sources.
  // No backend, so it must not point at one; the build rewrites those links to
  // placeholders that preview.js fills from what it actually has.
  assert.ok(!preview.includes('/api/'), 'the built preview still references the studio API');
  const external = ['/artifacts/'];
  const urls = [...preview.matchAll(/(?:src|href)="([^"]+)"/g)].map((match) => match[1]);
  assert.ok(urls.length, 'the built preview references no assets at all');
  for (const url of urls) {
    // Fragments cannot miss, external citations are not ours to serve, and the package
    // is the host's job. Everything else must be a file this build produced.
    if (url.startsWith('#') || /^https?:\/\//.test(url)) continue;
    if (external.some((prefix) => url.startsWith(prefix))) continue;
    assert.ok(url.startsWith('/'), `the built preview references a relative URL: ${url}`);
    assert.ok(existsSync(new URL(`.${url}`, dist)),
      `the built preview references ${url}, absent from web/dist`);
  }
  console.log(`Built pages: 2 entries bundled; ${urls.length} asset URLs in preview.html all resolve.`);
}
