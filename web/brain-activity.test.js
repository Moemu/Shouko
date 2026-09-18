import assert from 'node:assert/strict';
import { activityDisplay } from './brain-activity.js';

const frame = (time, activity, episode = 1) => ({ time, activity, episode });
const previous = frame(1, [0.2, 0.2, 0.2]);
const current = frame(1.1, [0.225, 0.175, 0.2]);
const change = activityDisplay(current, previous, 'change');
assert.ok(Math.abs(change.signals[0] - 0.5) < 1e-6);
assert.ok(Math.abs(change.signals[1] + 0.5) < 1e-6);
assert.equal(change.signals[2], 0);
assert.ok(Math.abs(change.rms - Math.sqrt(0.125 / 3)) < 1e-6);
for (const reset of [frame(1.1, [0.9, 0.9, 0.9], 2), frame(1, [0.9, 0.9, 0.9]), frame(2, [0.9, 0.9, 0.9])]) {
  assert.deepEqual([...activityDisplay(reset, previous, 'change').signals], [0, 0, 0]);
}
assert.deepEqual([...activityDisplay(previous, null, 'change').signals], [0, 0, 0]);
assert.deepEqual([...activityDisplay(frame(1.1, [2, -2, 0]), previous, 'strength').signals], [1, -1, 0]);
console.log('Activity display: fixed scales, signed changes, resets and gaps verified.');
