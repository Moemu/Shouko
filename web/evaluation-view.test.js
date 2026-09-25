// Covers the contract the static preview broke: the results content is collapsed by
// default, and expanding the details must reveal what was rendered. The visibility was
// previously owned by a listener that lived outside the renderer, so moving the renderer
// to a shared module silently left the page with a report that never appeared.
import assert from 'node:assert/strict';
import { t } from './i18n.js';

const elements = new Map();
const makeElement = (id) => ({
  id, textContent: '', innerHTML: '', hidden: false, open: false,
  dataset: {}, listeners: {},
  addEventListener(type, handler) { (this.listeners[type] ||= []).push(handler); },
  fire(type) { for (const handler of this.listeners[type] || []) handler(); },
});
globalThis.document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, makeElement(id));
    return elements.get(id);
  },
};

const { renderEvaluation, setEvaluationState } = await import('./evaluation-view.js');
const el = (id) => document.getElementById(id);

const walk = (seed, overrides = {}) => ({
  seed, condition: null, target_speed: 0.5, mean_speed: 0.48, seconds: 30,
  foot_strikes: [7, 8], alternations: 12, contact_fraction: [0.8, 0.85],
  minimum_upright: 0.9, inference_ms_median: 7.5, inference_ms_p95: 14.7,
  criteria: { survived: true, speed: true, both_feet: true, alternation: true, upright: true, direction: true },
  success: true, fallen: false, ...overrides,
});
const record = {
  kind: 'held_out_native_mujoco', complete: true, robot: 'yumi',
  checkpoint_sha256: 'f1a20071'.padEnd(64, '0'), successes: 2, attempts: 2,
  tests: [walk(1), walk(2, { mean_speed: 0.2, success: false, fallen: true })],
};

renderEvaluation(record, { meta: { mode: 'full_connectome' } });
assert.equal(el('evaluationState').textContent, t('eval.done'), 'the pill reports a matched report');
assert.match(el('evaluationRows').innerHTML, /group-row/, 'the rows were rendered');
assert.equal(el('evaluationContent').hidden, true, 'content stays collapsed until the details are open');

el('evaluationDetails').open = true;
el('evaluationDetails').fire('toggle');
assert.equal(el('evaluationContent').hidden, false, 'expanding the details reveals the content');

// Collapsing again must hide it, and a state message must clear both panes.
el('evaluationDetails').open = false;
el('evaluationDetails').fire('toggle');
assert.equal(el('evaluationContent').hidden, true);
setEvaluationState('absent', t('eval.absent'));
assert.equal(el('evaluationContent').hidden, true);
assert.equal(el('evaluationSummary').innerHTML, '');
assert.equal(el('evaluationRows').textContent, '');

console.log('Evaluation view: collapsed by default, expanding reveals the rendered report.');
