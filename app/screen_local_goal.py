"""Predeclared three-seed local exploration by work-cost factorial campaign."""
import argparse
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .diagnose_goal_d0 import write_json
from .evaluate_goal import file_sha256
from .evaluate_local_goal import evaluate, passes_gate
from .free_body import ROOT
from .local_goal import prepare_state


@dataclass(frozen=True)
class Arm:
    name: str
    rho: float
    work_cost: float


ARMS = (Arm('iid_cost', 0., .001), Arm('iid_zero', 0., 0.),
        Arm('persistent_cost', .9, .001), Arm('persistent_zero', .9, 0.))
ORDER = ((0,1,2,3),(2,3,0,1),(1,0,3,2))
SOURCES = ('app/local_goal.py','app/goal_checkpoint.py','app/ppo_goal.py','app/ppo_local_goal.py',
           'app/evaluate_local_goal.py','app/screen_local_goal.py','app/free_body.py','app/gpu_goal.py',
           'app/goal_task.py','app/mlp_policy.py','app/ppo_yumi.py','app/recovery_curriculum.py','app/value_diagnostics.py')


def run(output: Path, max_seconds: float) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = started + max_seconds
    state_path = output / 'reset.npz'
    state_hash = prepare_state(state_path)
    sources = {name:file_sha256(ROOT/name) for name in SOURCES}
    for name in SOURCES:
        snapshot = output/'source_snapshot'/name
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes((ROOT/name).read_bytes())
    write_json(output/'preregistration.json', dict(arms=[asdict(a) for a in ARMS],seeds=[1401,1402,1403],
               run_order=ORDER,iterations=256,worlds=16,steps=64,initial_std=.7,transitions_per_run=262144,
               per_run_seconds=420,training_process_seconds=5400,max_wall_seconds=max_seconds,
               started_unix=started,deadline_unix=deadline,state_sha256=state_hash,source_sha256=sources))
    rows = []
    training_seconds = 0.
    branch = ''
    for seed, order in zip((1401,1402,1403),ORDER):
        reference = None
        initial_evals = {}
        for index in order:
            arm = ARMS[index]
            if time.time() >= deadline or training_seconds >= 5400:
                branch = 'budget_stop'; break
            assert sources == {name:file_sha256(ROOT/name) for name in SOURCES}, 'Source changed during campaign'
            name = f'{arm.name}_{seed}'
            folder = output/name
            write_json(output/'campaign.json',dict(phase='training',current=name,runs=rows,training_seconds=training_seconds))
            command = [sys.executable,'-m','app.ppo_local_goal','--output',str(folder),'--state',str(state_path),
                       '--seed',str(seed),'--rho',str(arm.rho),'--work-cost',str(arm.work_cost),
                       '--max-seconds',str(min(420.,deadline-time.time(),5400-training_seconds))]
            launch = time.perf_counter()
            with (output/(name+'.log')).open('w',encoding='utf-8') as log:
                process = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            training_seconds += time.perf_counter()-launch
            row = dict(group=arm.name,seed=seed,rho=arm.rho,work_cost=arm.work_cost,passed=False,complete=False)
            if process.returncode:
                status = json.loads((folder/'training.json').read_text(encoding='utf-8')) if (folder/'training.json').exists() else {}
                row.update(error=status.get('error','Training process failed'),physical_guard=status.get('physical_guard'),
                           attempted_transitions=status.get('attempted_local_transitions'),actual_transitions=status.get('env_steps',0))
                rows.append(row)
                write_json(output/'campaign.json',dict(phase='recorded_failure',runs=rows,training_seconds=training_seconds))
                if not status.get('physical_guard'):
                    branch='engineering_block'; break
                continue
            training = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            row['actual_transitions'] = training['launch_transitions']
            if training['launch_transitions'] != 262144:
                row['error']='Incomplete fixed sample budget';rows.append(row);branch='budget_stop';break
            initial = torch.load(folder/'initial.pt',map_location='cpu',weights_only=True)
            if reference is None:
                reference = initial
            for section in ('policy','value'):
                assert all(torch.equal(t,initial[section][k]) for k,t in reference[section].items()), (name,section)
            assert torch.equal(reference['log_std'],initial['log_std'])
            if arm.rho not in initial_evals:
                initial_evals[arm.rho] = {backend:evaluate(folder/'initial.pt',state_path,
                    output/f'initial_{name}_{backend}',backend) for backend in ('native','cuda')}
            row['initial'] = initial_evals[arm.rho]
            row['final'] = {backend:evaluate(folder/'last.pt',state_path,output/f'eval_{name}_{backend}',backend)
                            for backend in ('native','cuda')}
            row['backend_passed'] = {backend:passes_gate(row['final'][backend],row['initial'][backend])
                                     for backend in ('native','cuda')}
            row['passed'] = all(row['backend_passed'].values())
            row['complete'] = True
            row['checkpoint_sha256'] = training['checkpoint_sha256']
            rows.append(row)
            write_json(output/'campaign.json',dict(phase='evaluated',runs=rows,training_seconds=training_seconds))
            print(json.dumps(dict(run=name,passed=row['passed'],native_progress=row['final']['native']['net_progress_m'],
                                  cuda_progress=row['final']['cuda']['net_progress_m'])),flush=True)
        if branch:
            break
    groups = []
    for arm in ARMS:
        completed = [r for r in rows if r['group']==arm.name and r['complete']]
        group = dict(group=arm.name,rho=arm.rho,work_cost=arm.work_cost,complete_runs=len(completed),
                     passed_seeds=sum(r['passed'] for r in completed),
                     arrivals=sum(bool(r['final']['native']['reached']) for r in completed))
        for metric in ('net_progress_m','after_3s_progress_m','common_reward','reached'):
            group[metric] = float(np.mean([r['final']['native'][metric] for r in completed])) if completed else None
        groups.append(group)
    selected = None
    if not branch:
        candidates = [g for g in groups if g['passed_seeds']>=2]
        preference = {'iid_cost':0,'persistent_cost':1,'iid_zero':2,'persistent_zero':3}
        candidates.sort(key=lambda g:(-g['passed_seeds'],-g['arrivals'],-g['after_3s_progress_m'],preference[g['group']]))
        selected = candidates[0]['group'] if candidates else None
        branch = 'local_learning_positive' if selected else 'route_uncertainty_stop'
        if candidates and not any(g['work_cost']>0 for g in candidates) and candidates[0]['common_reward']<=0:
            branch = 'local_motion_cost_tradeoff'
    report = dict(phase='complete' if len(rows)==12 else 'stopped',next_branch=branch,selected_group=selected,
                  runs=rows,groups=groups,actual_transitions=sum(r.get('actual_transitions',0) for r in rows),
                  full_factorial=len(rows)==12 and all(r['complete'] for r in rows),
                  training_process_seconds=training_seconds,elapsed_seconds=time.time()-started,
                  source_sha256=sources,state_sha256=state_hash,preregistration_sha256=file_sha256(output/'preregistration.json'))
    write_json(output/'summary.json',report)
    print(json.dumps({k:report[k] for k in ('next_branch','selected_group','actual_transitions','elapsed_seconds')}),flush=True)
    return report


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-seconds',type=float,default=10800.)
    args=parser.parse_args()
    if not 0<args.max_seconds<=10800:
        raise ValueError('Use a wall budget in (0,10800] seconds')
    run(args.output,args.max_seconds)


if __name__=='__main__':
    main()
