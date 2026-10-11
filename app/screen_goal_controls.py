"""Fixed three-arm, three-seed local screen; retain every final checkpoint and result."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .evaluate_goal import fixed_deck, file_sha256
from .free_body import ROOT
from .train_full import atomic_json


SEEDS = (404, 505, 606)
GROUPS = ('baseline', 'no_work_cost', 'exploration')


def passes_gate(row: dict) -> bool:
    return (row['mean_first_target_progress_m'] > 0
            and row['paired_progress_improvement_m'] >= .10 - 1e-9
            and row['mean_after_3s_first_target_progress_m'] >= .10 - 1e-9
            and row['mean_first_target_min_distance_m'] <= 1.5 + 1e-9
            and row['max_penetration_m'] <= .030
            and row['max_joint_limit_error_rad'] <= .15)


def run_child(module: str, arguments: list[str], console: Path) -> None:
    with console.open('w', encoding='utf-8') as handle:
        completed = subprocess.run([sys.executable, '-m', module, *arguments], cwd=ROOT,
                                   stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f'{module} failed with exit code {completed.returncode}; see {console.name}')


def analyze_run(output: Path, group: str, seed: int) -> dict:
    folder = output / f'{group}_{seed}'
    training = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    report = json.loads((output / f'eval_{group}_{seed}/summary.json').read_text(encoding='utf-8'))
    initial = json.loads((output / f'eval_initial_{seed}/summary.json').read_text(encoding='utf-8'))
    assert training['launch_transitions'] == 262144 and training['initial_env_steps'] == 0
    assert report['checkpoint_sha256'] == file_sha256(folder / 'last.pt') == training['checkpoint_sha256']
    assert report['deck_sha256'] == initial['deck_sha256'] == file_sha256(output / 'deck.json')
    assert report['contract'] == training['contract'] == initial['contract']
    trajectories = np.load(output / f'eval_{group}_{seed}/trajectories.npz')
    late_progress = []
    for episode in report['episodes']:
        trace = trajectories[f"episode_{episode['id']}"]
        first = trace[trace[:, 1] == 0]
        if first[-1, 0] <= 3.:
            late_progress.append(0.)
        else:
            at_three = first[np.argmin(np.abs(first[:, 0] - 3.))]
            assert abs(at_three[0] - 3.) < 1e-6
            distance_at_three = float(np.linalg.norm(at_three[8:10] - at_three[2:4]))
            late_progress.append(distance_at_three - episode['targets'][0]['final_distance_m'])
    progress = float(np.mean([e['targets'][0]['net_progress_m'] for e in report['episodes']]))
    initial_progress = float(np.mean([e['targets'][0]['net_progress_m'] for e in initial['episodes']]))
    row = dict(group=group, seed=seed, transitions=training['launch_transitions'],
               checkpoint_sha256=report['checkpoint_sha256'], settings=training['settings'],
               training_arrivals=training['arrivals'], training_completions=training['completed_episodes'],
               training_seconds=training['elapsed_seconds'],
               training_samples_per_second=training['training_samples_per_second'],
               actor_optimizer_steps=training['actor_optimizer_steps'],
               arrivals=report['arrivals'], first_target_arrivals=report['first_target_arrivals'],
               complete_episodes=report['complete_episodes'], termination_reasons=report['termination_reasons'],
               mean_first_target_progress_m=progress, paired_progress_improvement_m=progress - initial_progress,
               mean_after_3s_first_target_progress_m=float(np.mean(late_progress)),
               mean_first_target_min_distance_m=report['mean_first_target_min_distance_m'],
               mean_work_j=report['mean_work_j'], mean_speed_m_s=report['mean_speed_m_s'],
               max_penetration_m=report['max_penetration_m'],
               max_joint_limit_error_rad=max(e['max_joint_limit_error_rad'] for e in report['episodes']))
    row['gate_passed'] = passes_gate(row)
    return row


def screen(output: Path, exploration_std: float) -> dict:
    if exploration_std not in (.5, .7):
        raise ValueError('Use a preregistered exploration probe candidate: 0.5 or 0.7')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'deck.json').write_text(json.dumps(fixed_deck(), indent=2) + '\n', encoding='utf-8')
    settings = dict(baseline=dict(std=.35, work_cost=.001), no_work_cost=dict(std=.35, work_cost=0.),
                    exploration=dict(std=exploration_std, work_cost=.001))
    preregistration = dict(seeds=SEEDS, groups=settings, worlds=16, steps=64, iterations=256,
                          transitions_per_run=262144, max_training_seconds_per_run=420,
                          deck_sha256=file_sha256(output / 'deck.json'), diagnostic_deck_previously_viewed=True)
    atomic_json(output / 'preregistration.json', preregistration)
    started = time.perf_counter()
    rows = []
    training_seconds = 0.
    state = dict(phase='running', current_run=None, completed_runs=0, rows=[])
    atomic_json(output / 'campaign.json', state)
    try:
        for seed in SEEDS:
            for group in GROUPS:
                if training_seconds >= 7200 or time.perf_counter() - started >= 14400:
                    raise TimeoutError('Local campaign budget exhausted')
                state['current_run'] = f'{group}_{seed}'
                state['stage'] = 'training'
                atomic_json(output / 'campaign.json', state)
                folder = output / f'{group}_{seed}'
                config = settings[group]
                run_child('app.ppo_goal', ['--output', str(folder), '--seed', str(seed), '--worlds', '16',
                          '--steps', '64', '--iterations', '256', '--max-seconds', '420',
                          '--initial-std', str(config['std']), '--work-cost', str(config['work_cost'])],
                          output / f'train_{group}_{seed}.log')
                training = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
                if training['launch_transitions'] != 262144:
                    raise RuntimeError('Incomplete sample budget; do not selectively extend the run')
                training_seconds += training['elapsed_seconds']
                initial = torch.load(folder / 'initial.pt', map_location='cpu', weights_only=True)
                if group == 'baseline':
                    reference = initial['policy']
                    state['stage'] = 'initial_evaluation'
                    atomic_json(output / 'campaign.json', state)
                    run_child('app.evaluate_goal', ['--deck', str(output / 'deck.json'), '--checkpoint', str(folder / 'initial.pt'),
                              '--output', str(output / f'eval_initial_{seed}')], output / f'eval_initial_{seed}.log')
                else:
                    for name in reference:
                        torch.testing.assert_close(initial['policy'][name], reference[name], rtol=0, atol=0)
                state['stage'] = 'final_evaluation'
                atomic_json(output / 'campaign.json', state)
                run_child('app.evaluate_goal', ['--deck', str(output / 'deck.json'), '--checkpoint', str(folder / 'last.pt'),
                          '--output', str(output / f'eval_{group}_{seed}')], output / f'eval_{group}_{seed}.log')
                row = analyze_run(output, group, seed)
                rows.append(row)
                state.update(completed_runs=len(rows), rows=rows, training_seconds=training_seconds)
                atomic_json(output / 'campaign.json', state)
                print(json.dumps(row, allow_nan=False), flush=True)
        candidates = []
        for group in GROUPS:
            selected = [row for row in rows if row['group'] == group]
            passed = sum(row['gate_passed'] for row in selected)
            candidates.append(dict(group=group, passed_seeds=passed,
                                   first_target_arrivals=sum(row['first_target_arrivals'] for row in selected),
                                   mean_first_target_progress_m=float(np.mean([row['mean_first_target_progress_m'] for row in selected]))))
        priority = {'baseline': 0, 'exploration': 1, 'no_work_cost': 2}
        qualified = [item for item in candidates if item['passed_seeds'] >= 2]
        qualified.sort(key=lambda item: (-item['passed_seeds'], -item['first_target_arrivals'],
                                        -item['mean_first_target_progress_m'], priority[item['group']]))
        selected_group = qualified[0]['group'] if qualified else None
        state.update(phase='complete', stage='decision', current_run=None, candidates=candidates,
                     selected_group=selected_group,
                     next_branch='extended_verification' if selected_group else 'route_uncertainty_stop',
                     elapsed_seconds=time.perf_counter() - started,
                     preregistration=preregistration, initial_policies_equal=True)
        names = ('app/screen_goal_controls.py', 'app/ppo_goal.py', 'app/evaluate_goal.py', 'app/goal_checkpoint.py')
        state['source_sha256'] = {name: file_sha256(ROOT / name) for name in names}
        atomic_json(output / 'summary.json', state)
        atomic_json(output / 'campaign.json', state)
        return state
    except Exception as error:
        state.update(phase='failed', error=f'{type(error).__name__}: {error}')
        atomic_json(output / 'campaign.json', state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exploration-std', type=float, required=True)
    args = parser.parse_args()
    screen(args.output.resolve(), args.exploration_std)


if __name__ == '__main__':
    main()
