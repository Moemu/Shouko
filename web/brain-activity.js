/** Fixed display scales: signed model activity, or activity change per simulation second. */
export function activityDisplay(current, previous, mode) {
  const dt = previous ? current.time - previous.time : 0;
  const comparable = previous && current.episode === previous.episode
    && current.activity.length === previous.activity.length && dt > 0 && dt <= 0.5;
  const values = Float32Array.from(current.activity, (value, index) => mode === 'change'
    ? comparable ? (value - previous.activity[index]) / dt : 0
    : value);
  const scale = mode === 'change' ? 0.5 : 1;
  const rms = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / Math.max(1, values.length));
  return { signals: values.map(value => Math.max(-1, Math.min(1, value / scale))), rms };
}
