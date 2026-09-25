// Scratch: CDP interaction smoke test on the visible Edge preview tab.
const list = await (await fetch('http://127.0.0.1:9333/json/list')).json();
const page = list.find(t => t.type === 'page' && t.url.includes('preview.html') && !t.url.includes('old'));
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0; const pending = new Map();
const send = (method, params = {}) => new Promise((res) => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({id: i, method, params})); });
ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); } };
await new Promise(r => { ws.onopen = r; });
const evalJs = async (expr) => (await send('Runtime.evaluate', {expression: expr, awaitPromise: true, returnByValue: true})).result?.value;
await send('Page.enable');
await send('Page.bringToFront');
await new Promise(r => setTimeout(r, 300));

const out = {};
out.push = await evalJs(`(async () => {
  document.getElementById('pushButton').click();
  await new Promise(r => setTimeout(r, 1200));
  const l = window.__preview;
  return {height: +l.body.data.qpos[2].toFixed(3), upright: +(-l.body._gravity[2]).toFixed(3), running: l.running};
})()`);
out.speed = await evalJs(`(async () => {
  const s = document.getElementById('targetSpeed');
  s.value = 0.25; s.dispatchEvent(new Event('input')); s.dispatchEvent(new Event('change'));
  await new Promise(r => setTimeout(r, 2500));
  const l = window.__preview;
  return {loopSpeed: l.speed, vNow: +l.body.data.qvel[0].toFixed(3), rtf: +l.rtf.toFixed(2)};
})()`);
out.lesion = await evalJs(`(async () => {
  const s = document.getElementById('lesion');
  s.value = 'disconnected'; s.dispatchEvent(new Event('change'));
  await new Promise(r => setTimeout(r, 1500));
  const a = window.__preview.outputRms;
  s.value = 'intact'; s.dispatchEvent(new Event('change'));
  await new Promise(r => setTimeout(r, 1500));
  return {lesionOutputRms: +a.toFixed(3), intactOutputRms: +window.__preview.outputRms.toFixed(3)};
})()`);
out.lesionEpisode = await evalJs(`(async () => {
  const l = window.__preview;
  const before = l.episode;
  const sel = document.getElementById('lesion');
  sel.value = 'disconnected'; sel.dispatchEvent(new Event('change'));
  await new Promise(r => setTimeout(r, 800));
  const afterDisc = {episode: l.episode, lesion: l.lesion, playLabel: document.getElementById('playLabel').textContent};
  sel.value = 'intact'; sel.dispatchEvent(new Event('change'));
  await new Promise(r => setTimeout(r, 800));
  return {before, afterDisc, afterIntact: {episode: l.episode, lesion: l.lesion}};
})()`);
out.strengthLabels = await evalJs(`(async () => {
  document.getElementById('brainstrength').click();
  await new Promise(r => setTimeout(r, 300));
  const r = {scale: document.getElementById('brainScale').textContent, label: document.getElementById('activityLabel').textContent};
  document.getElementById('brainchange').click();
  await new Promise(r => setTimeout(r, 300));
  r.scaleBack = document.getElementById('brainScale').textContent;
  return r;
})()`);
out.gpuErrors = await evalJs(`window.__preview.brain.errors.length`);
out.speedText = await evalJs(`document.getElementById('speed').textContent`);
console.log(JSON.stringify(out, null, 2));
process.exit(0);
