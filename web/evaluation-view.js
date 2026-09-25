// Renders a held-out evaluation record into the read-only results section.
//
// The studio fills this from /api/evaluation, which only serves evidence bound to
// the loaded checkpoint; the static preview fills it from its package and checks the
// same binding itself before calling in. Both share one renderer, so the record is a
// parameter rather than module state — neither page has to own the other's bookkeeping.
import { t } from './i18n.js';
import { speedError } from './training-series.js';
import { $, setText, setHTML, setKey, escapeHTML, number } from './dom.js';

/** `state` is 'absent' | 'failed': an empty evaluation collapses quietly, a real
    failure stays visible even while the details stay collapsed. */
export function setEvaluationState(state, message, { running = false } = {}) {
  setText($('evaluationState'), running && state === 'absent' ? t('eval.running') : message);
  $('evaluationState').dataset.failed = String(state === 'failed');
  $('evaluationError').hidden = state !== 'failed';
  if (state === 'failed') setText($('evaluationError'), message);
  $('evaluationContent').hidden = true;
  setHTML($('evaluationSummary'), ''); setText($('evaluationRows'), '');
}

// The content only shows while the details are open, and the reader can toggle them
// before or after a record arrives. Binding it here rather than in each page keeps the
// renderer's visibility contract whole — a page that renders gets the toggle with it.
let toggleBound = false;
function bindToggleOnce() {
  if (toggleBound) return;
  toggleBound = true;
  $('evaluationDetails').addEventListener('toggle', () => {
    $('evaluationContent').hidden = !$('evaluationDetails').open;
  });
}

export function renderEvaluation(record, { meta = null, running = false } = {}) {
  const data = record;
  if (!data) return setEvaluationState('absent', t('results.pending'), { running });
  bindToggleOnce();
  // While a job is active the phase pill is owned by the studio's status poll; the
  // record's own state only shows when nothing is running, so the pill never flickers.
  if (!running) {
    setText($('evaluationState'), t(data.complete === false ? 'eval.running' : 'eval.done'));
    $('evaluationState').dataset.failed = 'false';
  }
  $('evaluationError').hidden = true;
  $('evaluationContent').hidden = !$('evaluationDetails').open;
  const tests = data.tests || [], longWalks = data.long_walks || [], perturbations = data.perturbations || [];
  const controls = data.controls || {};
  const disconnected = Array.isArray(controls) ? controls : controls.disconnected || [];
  const rewired = Array.isArray(controls) ? [] : controls.rewired || [];
  setHTML($('evaluationSummary'),
    `<div><b>${escapeHTML(data.successes ?? '--')} / ${escapeHTML(data.attempts ?? '--')}</b><span>${t('results.summary.walk')}${data.complete === false ? t('results.summary.inProgress') : ''}</span></div>`
    + `<div><b>${escapeHTML(data.robot || '--')}</b><span>${t('eval.robot')}</span></div>`
    + `<div><b title="${escapeHTML(data.checkpoint_sha256 || '')}">${escapeHTML((data.checkpoint_sha256 || '').slice(0, 10) || '--')}</b><span>${t('eval.checkpoint')}</span></div>`);
  const strictGait = data.kind === 'held_out_locomotion';
  setKey($('resultsIntro'), strictGait ? 'results.pGait' : data.kind === 'held_out_native_mujoco' ? 'results.pFull' : meta?.mode === 'full_connectome' ? 'results.pFullScreening' : 'results.p');
  if (strictGait) setText($('resultsCriteria'), t('results.microGait', { lateral: number(data.long_lateral_m, 2), recovered: data.yaw_recovered, total: data.yaw_tests?.length || 0 }));
  else setKey($('resultsCriteria'), data.kind === 'held_out_native_mujoco' ? 'results.microFull' : meta?.mode === 'full_connectome' ? 'results.microFullScreening' : 'results.micro');
  const groups = [
    { key: 'walk', rows: tests, expect: 'pass' },
    { key: 'yaw', rows: data.yaw_tests || [], expect: 'pass' },
    { key: 'long', rows: longWalks, expect: 'pass' },
    { key: 'push', rows: perturbations, expect: 'pass' },
    { key: 'disconnected', rows: disconnected, expect: 'fall' },
    { key: 'rewired', rows: rewired, expect: 'fall' },
  ].filter(g => g.rows.length);
  const failed = ['survived', 'speed', 'both_feet', 'alternation', 'upright', 'direction', ...(strictGait ? ['swing'] : [])];
  setHTML($('evaluationRows'), groups.map(g => {
    const rows = g.rows.map(r => {
      const reason = r.criteria ? failed.filter(k => !r.criteria[k]).map(k => t(`results.fail.${k}`)).join(' / ') : t('results.noTarget');
      const ok = r.success || (g.expect === 'fall' && r.fallen);
      const cls = r.success ? 'pass' : ok ? 'neutral' : 'fail';
      const error = speedError(r);
      return `<tr><td>${escapeHTML(r.seed)} / ${escapeHTML(r.condition || t(`results.condition.${g.key}`))}</td><td>${number(r.target_speed, 2)} m/s</td>`
        + `<td>${number(r.mean_speed, 3)} m/s</td><td>${number(error, 3)}</td>`
        + `<td>${number(r.seconds, 1)} s</td><td>${number(r.foot_strikes?.[0], 0)} / ${number(r.foot_strikes?.[1], 0)}</td>`
        + `<td>${number(r.alternations, 0)}</td><td>${(r.contact_fraction || []).map(f => number(f, 2)).join(' / ') || '--'}</td>`
        + `<td>${number(r.minimum_upright, 2)}</td><td>${number(r.inference_ms_median, 2)} / ${number(r.inference_ms_p95, 2)} ms</td>`
        + `<td class="${cls}">${r.success ? t('results.pass') : r.fallen ? (g.expect === 'fall' ? t('results.expect.fall') : t('results.fallen')) : reason}</td></tr>`;
    }).join('');
    return `<tr class="group-row"><td colspan="11">${t(`results.group.${g.key}`)} · ${g.rows.filter(r => g.expect === 'fall' ? r.fallen : r.success).length} / ${g.rows.length}</td></tr>${rows}`;
  }).join(''));
  if (!groups.length) setHTML($('evaluationRows'), `<tr><td colspan="11">${t('results.pending')}</td></tr>`);
}
