"""Screen continuation gate boundaries, independent of training reward values."""
from .screen_goal_controls import passes_gate


def main() -> None:
    row = dict(mean_first_target_progress_m=.2, paired_progress_improvement_m=.1,
               mean_after_3s_first_target_progress_m=.1, mean_first_target_min_distance_m=1.5,
               max_penetration_m=.03, max_joint_limit_error_rad=.15)
    assert passes_gate(row)
    failures = dict(mean_first_target_progress_m=0., paired_progress_improvement_m=.099,
                    mean_after_3s_first_target_progress_m=.099, mean_first_target_min_distance_m=1.501,
                    max_penetration_m=.0301, max_joint_limit_error_rad=.1501)
    for name, value in failures.items():
        assert not passes_gate(dict(row, **{name: value})), name
    assert passes_gate(dict(row, irrelevant_training_reward=-100.))
    print('PASS: fixed goal screen continuation gates')


if __name__ == '__main__':
    main()
