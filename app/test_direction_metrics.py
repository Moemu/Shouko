"""Coordinate decomposition and post-event timing regression checks."""
import numpy as np
from .direction_metrics import direction_summary, sustained_time


def main():
    time = np.arange(0, 12.001, .02)
    yaw = np.full_like(time, .2)
    positions = np.column_stack([.5*np.cos(.2)*time, .5*np.sin(.2)*time])
    summary = direction_summary(time, positions, yaw)
    assert abs(summary['body_lateral_y_m']) < 1e-10
    assert abs(summary['heading_y_m'] - positions[-1, 1]) < 1e-10
    positions[:, 1] = -.03*time
    summary = direction_summary(time, positions, np.zeros_like(time))
    assert abs(summary['steady_body_lateral_mps'] + .03) < 1e-10
    assert abs(summary['decomposition_residual_m']) < 1e-10
    values = np.zeros_like(time)
    values[(time >= 10) & (time < 11)] = .2
    assert sustained_time(time, values, .05, .5, 10) >= 1
    assert sustained_time(time, np.full_like(time, .2), .05, .5, 10) is None
    assert sustained_time(time[:3], values[:3], .05, .5, 10) is None
    print('Direction decomposition and event timing checks passed')


if __name__ == '__main__':
    main()
