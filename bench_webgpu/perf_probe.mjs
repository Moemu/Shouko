// Scratch perf probe: attach to the visible Edge preview tab via CDP and
// sample rAF gaps + long tasks. Deletable with the rest of bench_webgpu/.
const list = await (await fetch('http://127.0.0.1:9333/json/list')).json();
const want = process.argv[2] || 'preview.html';
const page = list.find(t => t.type === 'page' && t.url.includes('preview'));
if (!page) { console.error('no preview tab'); process.exit(1); }
if (!page) { console.error('page not found:', list.map(t => `${t.type} ${t.url}`)); process.exit(1); }
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0; const pending = new Map();
const send = (method, params = {}) => new Promise((res) => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({id: i, method, params})); });
ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); } };
await new Promise(r => { ws.onopen = r; });
const evalJs = async (expr) => (await send('Runtime.evaluate', {expression: expr, awaitPromise: true, returnByValue: true})).result?.value;

await send('Page.enable');
if (!page.url.includes(want)) { await send('Page.navigate', {url: 'http://127.0.0.1:8899/web/' + want}); await new Promise(r => setTimeout(r, 15000)); }
await send('Page.bringToFront');
await new Promise(r => setTimeout(r, 500));
const vis0 = await evalJs('document.visibilityState');

await evalJs(`(() => {
  window.__perf = {longTasks: [], gaps: [], frames: 0};
  new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__perf.longTasks.push(Math.round(e.duration)); }).observe({entryTypes: ['longtask']});
  let last = performance.now();
  const tick = () => { const now = performance.now(); window.__perf.gaps.push(+(now - last).toFixed(1)); last = now; if (window.__perf.frames++ < 900) requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
  return 'ok';
})()`);

await new Promise(r => setTimeout(r, 12000));

const out = await evalJs(`(() => {
  const p = window.__perf;
  const g = [...p.gaps].sort((a, b) => a - b);
  const lt = [...p.longTasks].sort((a, b) => a - b);
  const q = (a, f) => a.length ? a[Math.floor(a.length * f)] : null;
  const l = window.__preview;
  return {
    vis: document.visibilityState, vis0: 'sent', frames: p.frames,
    gapMed: q(g, 0.5), gapP90: q(g, 0.9), gapMax: g[g.length - 1],
    ltCount: p.longTasks.length, ltMed: q(lt, 0.5), ltMax: lt[lt.length - 1],
    brainMs: l && +l.brainMs.toFixed(1), rtf: l && +l.rtf.toFixed(2),
    running: l && l.running, simTime: l && +l.body.data.time.toFixed(1),
    gpuErrors: l && l.brain.errors && l.brain.errors.length,
    canvasW: document.querySelector('#viewport canvas')?.width,
    speedText: document.querySelector('#speed')?.textContent,
  };
})()`);
console.log(JSON.stringify(out, null, 2));

// Extra: quantify setSize cost on the live visible renderer via canvas reassign
const sizeCost = await evalJs(`(async () => {
  const c = document.querySelector('#viewport canvas');
  if (!c) return 'no canvas';
  c.width = 1238; c.height = 1058;
  await new Promise(r => requestAnimationFrame(r));
  const t0 = performance.now();
  for (let i = 0; i < 60; i++) { c.width = 1238; c.height = 1058; }
  const reassign = (performance.now() - t0) / 60;
  return {reassignMs: +reassign.toFixed(3)};
})()`);
console.log('sizeCost', JSON.stringify(sizeCost));
process.exit(0);
