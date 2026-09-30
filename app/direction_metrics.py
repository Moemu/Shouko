"""Direction diagnostics from native planar motion, separate from gait acceptance."""
import numpy as np


def sustained_time(times, values, threshold, duration, after=0.0):
    times, values = np.asarray(times), np.asarray(values)
    eligible = np.flatnonzero(times >= after - 1e-9)
    if len(eligible) < 2:
        return None
    dt = float(np.median(np.diff(times)))
    count = max(1, int(np.ceil(duration / dt - 1e-8)))
    valid = np.abs(values[eligible]) <= threshold
    if len(valid) < count:
        return None
    windows = np.convolve(valid.astype(int), np.ones(count, dtype=int), 'valid')
    hits = np.flatnonzero(windows == count)
    return float(times[eligible[hits[0]]] - after) if len(hits) else None


def direction_summary(times, positions, yaw, event_time=None, period=0.8):
    times, positions, yaw = map(np.asarray, (times, positions, yaw))
    dt = np.diff(times)
    velocity = np.diff(positions[:, :2], axis=0) / dt[:, None]
    angle = (np.unwrap(yaw)[1:] + np.unwrap(yaw)[:-1]) / 2
    forward = np.cos(angle)*velocity[:, 0] + np.sin(angle)*velocity[:, 1]
    lateral = -np.sin(angle)*velocity[:, 0] + np.cos(angle)*velocity[:, 1]
    heading_y = float(np.sum(forward*np.sin(angle)*dt))
    lateral_y = float(np.sum(lateral*np.cos(angle)*dt))
    steady = times[1:] >= 5
    if not steady.any():
        steady = np.ones_like(lateral, dtype=bool)
    cycles = []
    for start in np.arange(5.0, times[-1] - period + 1e-8, period):
        mask = (times[1:] >= start) & (times[1:] < start + period)
        if mask.any():
            cycles.append([float(lateral[mask].mean()), float(yaw[1:][mask].mean())])
    return dict(signed_end_y_m=float(positions[-1, 1]),
                maximum_lateral_m=float(np.abs(positions[:, 1]).max()),
                heading_y_m=heading_y, body_lateral_y_m=lateral_y,
                decomposition_residual_m=float(positions[-1, 1]-positions[0, 1]-heading_y-lateral_y),
                steady_body_lateral_mps=float(lateral[steady].mean()),
                steady_body_lateral_rms_mps=float(np.sqrt(np.mean(lateral[steady]**2))),
                cycle_body_lateral_abs_mean_mps=float(np.mean(np.abs(np.asarray(cycles)[:, 0]))) if cycles else None,
                cycle_yaw_abs_mean_rad=float(np.mean(np.abs(np.asarray(cycles)[:, 1]))) if cycles else None,
                yaw_rms_rad=float(np.sqrt(np.mean(yaw**2))),
                yaw_recovery_after_event_s=sustained_time(times, yaw, .05, .5, event_time) if event_time is not None else None,
                lateral_recovery_after_event_s=sustained_time(times[1:], lateral, .05, .5, event_time) if event_time is not None else None,
                limitation='Body lateral motion is not contact-point slip. Recovery is the first post-event window, not permanent recovery.')
