/** Real log fields only. Each metric gets its own scale; absent values stay absent. */
export const finite = value => typeof value === 'number' && Number.isFinite(value);
const metricFields = ['reward', 'pi_loss', 'v_loss', 'approx_kl', 'clipfrac', 'std', 'lr', 'mean_height', 'mean_knee', 'action_mse', 'validation_mse'];
export function trainingMetrics(history, algorithm) {
  const rows = (Array.isArray(history) ? history : []).filter(row => row && typeof row === 'object');
  const ppo = algorithm === 'ppo' || rows.some(row => finite(row.reward));
  const defaults = ppo ? ['reward', 'pi_loss', 'v_loss', 'approx_kl'] : ['action_mse'];
  // Metrics with no data at all are not rendered: an empty panel would
  // fabricate a series that the trainer never emitted.
  const shown = metricFields.filter(key => defaults.includes(key) || rows.some(row => finite(row[key])));
  return (shown.length ? shown : ['action_mse']).map(key => ({
    key, axis: ppo ? 'iteration' : 'samples',
    points: rows.map(row => [ppo ? row.iteration ?? row.samples : row.samples ?? row.update,
      finite(row[key]) ? row[key] : null, row.value_warmup === true]).filter(([x]) => finite(x)),
  }));
}
/** Kept for consumers of the original single-series helper. */
export function trainingSeries(history) {
  const rows = Array.isArray(history) ? history : [];
  const series = rows.filter(row => row && finite(row.samples ?? row.iteration) && finite(row.action_mse ?? row.validation_mse ?? row.reward))
    .map(row => [row.samples ?? row.iteration, row.action_mse ?? row.validation_mse ?? row.reward]);
  const kind = rows.some(row => row && finite(row.reward)) ? 'reward' : rows.some(row => row && finite(row.action_mse)) ? 'joint' : 'validation';
  return { series, kind, axis: kind === 'reward' ? 'iteration' : 'samples' };
}
export function trainingCapabilities(data, enabled, pending = false) {
  const active = data?.active === true;
  return { start: enabled === true && !!data && !active && !pending,
    pause: active && data.can_pause === true && !pending,
    resume: active && data.can_resume === true && !pending,
    stop: active && data.can_stop === true && !pending };
}
export function appendTrainingLog(previous, incoming, limit = 1000) {
  const changed = incoming.reset || incoming.run_id !== previous.run_id;
  const text = ((changed ? '' : previous.text || '') + (typeof incoming.text === 'string' ? incoming.text : '')).split('\n').slice(-limit).join('\n');
  return { run_id: incoming.run_id, cursor: incoming.cursor, text };
}
export function speedError(row) {
  return finite(row.mean_speed) && finite(row.target_speed) ? Math.abs(row.mean_speed - row.target_speed) : null;
}
