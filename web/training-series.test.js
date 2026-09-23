import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { trainingSeries, trainingMetrics, trainingCapabilities, appendTrainingLog, speedError } from './training-series.js';

/** The three real logs under runs/, if they are present on this machine. */
function realHistory(path) {
  if (!existsSync(path)) return null;
  return JSON.parse(readFileSync(path, 'utf8')).history;
}

// Synthetic shapes first, so the test still means something without the data files.
const subGraph = trainingSeries([{ samples: 0, validation_mse: 0.11 }, { samples: 512, validation_mse: 0.02 }]);
assert.deepEqual(subGraph.series, [[0, 0.11], [512, 0.02]]);
assert.equal(subGraph.kind, 'validation');
assert.equal(subGraph.axis, 'samples');

const fullGraph = trainingSeries([{ samples: 461056, action_mse: 0.0022 }]);
assert.deepEqual(fullGraph.series, [[461056, 0.0022]]);
assert.equal(fullGraph.kind, 'joint');

const ppo = trainingSeries([{ iteration: 1, reward: 2.62 }, { iteration: 2, reward: 3.1 }]);
assert.deepEqual(ppo.series, [[1, 2.62], [2, 3.1]]);
assert.equal(ppo.kind, 'reward');
assert.equal(ppo.axis, 'iteration');

// Rows that carry nothing usable are dropped, not plotted as NaN.
assert.deepEqual(trainingSeries([{ note: 'x' }, null, undefined, { samples: 1, reward: 0 }]).series, [[1, 0]]);
assert.deepEqual(trainingSeries(undefined).series, []);
assert.deepEqual(trainingSeries([]).series, []);

// Per-metric extraction: PPO defaults come first, extra real fields are added,
// missing values become gaps instead of invented zeros, warmup rows are flagged.
const ppoMetrics = trainingMetrics([
  { iteration: 1, reward: 2.5, pi_loss: 0.1, v_loss: 0.7, value_warmup: true },
  { iteration: 2, reward: null, pi_loss: 0.2, v_loss: 0.6 },
  { iteration: 3, reward: 3.1, pi_loss: 0.05, v_loss: 0.4 },
], 'ppo');
assert.deepEqual(ppoMetrics.map(m => m.key), ['reward', 'pi_loss', 'v_loss', 'approx_kl']);
const reward = ppoMetrics[0];
assert.deepEqual(reward.points, [[1, 2.5, true], [2, null, false], [3, 3.1, false]]);
assert.deepEqual(ppoMetrics[1].points, [[1, 0.1, true], [2, 0.2, false], [3, 0.05, false]]);
assert.deepEqual(trainingMetrics([{ samples: 0, validation_mse: 0.1 }], 'dagger').map(m => m.key), ['action_mse', 'validation_mse']);
// A history with none of the known fields falls back to the algorithm's
// defaults rather than rendering zero panels.
assert.deepEqual(trainingMetrics([{ iteration: 1 }], 'ppo').map(m => m.key), ['reward', 'pi_loss', 'v_loss', 'approx_kl']);
assert.deepEqual(trainingMetrics([{ samples: 1 }], 'dagger').map(m => m.key), ['action_mse']);
// A loss that is genuinely zero still plots; it is not a missing value.
const zeroMetrics = trainingMetrics([{ iteration: 1, reward: 0 }], 'ppo');
assert.deepEqual(zeroMetrics[0].points, [[1, 0, false]]);

// Capability matrix: state semantics, not wishful buttons.
const idle = trainingCapabilities({ active: false, can_stop: false }, true);
assert.deepEqual(idle, { start: true, pause: false, resume: false, stop: false });
const running = trainingCapabilities({ active: true, can_pause: true, can_resume: false, can_stop: true }, true);
assert.deepEqual(running, { start: false, pause: true, resume: false, stop: true });
assert.equal(trainingCapabilities({ active: true, can_stop: true }, true, true).stop, false, 'pending action must disable the row');
assert.equal(trainingCapabilities({ active: false }, false).start, false, 'meta-disabled environment');
assert.equal(trainingCapabilities({ active: true, can_pause: true }, true).resume, false);

// Log accumulation: bounded, run-scoped, reset-aware.
const base = { run_id: 'a', text: 'line1\nline2\n', cursor: 2 };
assert.equal(appendTrainingLog({ run_id: 'a', cursor: 0, text: '' }, base).text, 'line1\nline2\n');
assert.equal(appendTrainingLog(base, { run_id: 'a', text: 'line3\n', cursor: 3 }).text, 'line1\nline2\nline3\n');
assert.equal(appendTrainingLog(base, { run_id: 'a', reset: true, text: 'restarted\n', cursor: 1 }).text, 'restarted\n');
assert.equal(appendTrainingLog(base, { run_id: 'b', text: 'other\n', cursor: 1 }).text, 'other\n');
const manyLines = Array.from({ length: 1500 }, (_, i) => `l${i}`).join('\n');
const bounded = appendTrainingLog({ run_id: 'a', cursor: 0, text: '' }, { run_id: 'a', text: manyLines + '\n', cursor: 9 });
assert.equal(bounded.text.split('\n').length, 1000);
assert.equal(bounded.text.split('\n')[0], 'l501');

// Absolute speed error: measured minus target, never signed, never fabricated.
// 0.4-0.5 and 0.6-0.5 are not exactly representable in binary floats, so both
// compare with a tight tolerance; the sign must always come out positive.
assert.ok(speedError({ mean_speed: 0.4, target_speed: 0.5 }) > 0);
assert.ok(Math.abs(speedError({ mean_speed: 0.4, target_speed: 0.5 }) - 0.1) < 1e-9);
// 0.6-0.5 is not exactly representable; the function returns the raw difference,
// so compare with tolerance rather than pretending float magic.
assert.ok(Math.abs(speedError({ mean_speed: 0.6, target_speed: 0.5 }) - 0.1) < 1e-9);
assert.equal(speedError({ mean_speed: null, target_speed: 0.5 }), null);
assert.equal(speedError({}), null);

for (const path of ['runs/training.json', 'runs/cloud/training.json', 'runs/yumi/training.json']) {
  const history = realHistory(path);
  if (!history) continue;
  const { series, kind, axis } = trainingSeries(history);
  if (history.length === 0) {
    assert.equal(series.length, 0, 'A released ridge checkpoint must not inherit a PPO curve');
    continue;
  }
  assert.ok(series.length > 0, `${path} produced no plottable points`);
  for (const [x, y] of series) {
    assert.ok(Number.isFinite(x) && Number.isFinite(y), `${path} produced a non-finite point`);
  }
  console.log(`  ${path}: ${series.length} points, kind=${kind}, axis=${axis}`);
}

console.log('Training series: all three log formats produce finite points.');
