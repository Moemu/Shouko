"""Real step-boundary proof for finite deadlines, with legacy behavior retained."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .braking_task import FiniteLocalGoalTask, FiniteLocalGoalEnv, save_fork, restore_fork
from .diagnose_goal_d0 import write_json
from .free_body import FreeBody
from .local_goal import LocalGoalTask, LocalGoalEnv, LocalGoalSetup, prepare_state


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    state=args.output/'reset.npz';prepare_state(state)
    results=[]
    for finite in (False,True):
        task=(FiniteLocalGoalTask if finite else LocalGoalTask)(FreeBody(),state)
        env=(FiniteLocalGoalEnv if finite else LocalGoalEnv)(1,11,LocalGoalSetup(state,0.))
        task.target_elapsed=4.98;env.target_steps[:]=249
        _,native=task.step(np.zeros(18));_,cuda=env.step(torch.zeros(1,18,device='cuda'))
        assert native.reason=='timeout' and bool(cuda.timeout[0])
        assert native.terminated==finite and native.truncated!=finite
        assert bool(cuda.terminated[0])==finite and bool(cuda.truncated[0])!=finite
        assert task.done and bool(env.done[0])
        results.append(dict(finite=finite,native_terminated=native.terminated,cuda_terminated=bool(cuda.terminated[0])))
    task=FiniteLocalGoalTask(FreeBody(),state)
    for _ in range(5):task.step(np.full(18,.1))
    fork=args.output/'fork.npz';save_fork(task,fork)
    before=task.body.data.qvel.copy();previous=task.body.action.copy();elapsed=task.target_elapsed
    restored=FiniteLocalGoalTask(FreeBody(),fork);restore_fork(restored,fork,.2)
    np.testing.assert_array_equal(restored.body.data.qvel,before)
    np.testing.assert_array_equal(restored.body.action,previous)
    assert restored.target_elapsed==elapsed and abs(restored.target[1]-task.target[1]-.2)<1e-12
    assert restored.dwell==0 and restored.previous_distance==np.linalg.norm(restored.target-restored.body.center[:2])
    write_json(args.output/'checks.json',dict(passed=True,real_step_deadlines=results,
               full_velocity_and_previous_action_retained=True,target_change_does_not_extend_deadline=True))
    print('PASS: finite/legacy deadlines and complete replay fork')


if __name__=='__main__':main()
