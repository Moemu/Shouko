"""Owner-boundary checks for local reset, execution density, and checkpoint identity."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import torch

from .diagnose_goal_d0 import write_json
from .free_body import FreeBody
from .goal_checkpoint import goal_contract, save_goal_checkpoint, read_goal_payload, load_goal_policy
from .local_goal import LocalGoalSetup, LocalGoalTask, LocalGoalEnv, action_distribution, prepare_state
from .test_goal_training import training_parts, check_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    state = args.output / 'reset.npz'
    prepare_state(state)
    setup = LocalGoalSetup(state, .9)
    observation = torch.zeros(3, 69)
    observation[:, 48:66] = -1.
    distribution = action_distribution(torch.full((3,18), 2.), torch.full((18,), math.log(.7)), observation, .9)
    torch.testing.assert_close(distribution.mean, torch.full((3,18), -.7), atol=1e-7, rtol=0)
    expected_std = .7*math.sqrt(.19)
    torch.testing.assert_close(distribution.stddev, torch.full((3,18), expected_std), atol=1e-7, rtol=0)
    density = distribution.log_prob(torch.full((3,18), -.7)).sum(-1)
    np.testing.assert_allclose(density.numpy(), -18*math.log(expected_std*math.sqrt(2*math.pi)), rtol=0, atol=2e-6)
    legacy = check_checkpoint()
    actor, value, log_std, optimizer, value_optimizer = training_parts()
    checkpoint = args.output / 'local.pt'
    body = FreeBody()
    save_goal_checkpoint(checkpoint, contract=goal_contract(body), policy=actor, value=value, log_std=log_std,
                         optimizer=optimizer, value_optimizer=value_optimizer, iteration=0, env_steps=0,
                         settings={}, local_context=setup.context())
    read_goal_payload(checkpoint, goal_contract(body), local_context=setup.context())
    rejected = []
    for label, context in [('rho', replace(setup.context(), rho=0.)), ('state', replace(setup.context(), reset_sha256='0'*64))]:
        try:
            read_goal_payload(checkpoint, goal_contract(body), local_context=context)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError('Mismatched local context accepted')
    try:
        load_goal_policy(checkpoint, body)
    except ValueError:
        rejected.append('original_task_loader')
    else:
        raise AssertionError('Original task accepted local checkpoint')
    task = LocalGoalTask(body, state)
    env = LocalGoalEnv(2, 11, setup)
    np.testing.assert_allclose(env.observation()[0].cpu().numpy(), task.observation(), atol=2e-5, rtol=0)
    np.testing.assert_allclose(env.qvel[0].cpu().numpy(), body.data.qvel, atol=1e-7, rtol=0)
    max_qpos = 0.
    for _ in range(12):
        action = .15*np.sin(np.arange(18))
        task.step(action)
        env.step(torch.tensor(np.repeat(action[None],2,axis=0),dtype=torch.float32,device='cuda'))
        max_qpos = max(max_qpos, float(np.max(np.abs(env.qpos[0].cpu().numpy()-body.data.qpos))))
    assert max_qpos < 1e-3, max_qpos
    before = env.qpos[1].clone()
    mask = torch.tensor([True,False],device='cuda')
    env.reset(mask)
    torch.testing.assert_close(env.qpos[1], before, rtol=0,atol=0)
    task.reset()
    np.testing.assert_allclose(env.observation()[0].cpu().numpy(), task.observation(), atol=2e-5,rtol=0)
    full = torch.ones(2,dtype=torch.bool,device='cuda')
    env.reset(full)
    env.target_steps[:] = 249
    task.target_elapsed = 4.98
    native = task.evaluate(body.center, np.zeros(3), 0.)
    batched = env.evaluate(env.center.clone(), torch.zeros(2,3,device='cuda'), torch.zeros(2,device='cuda'))
    assert native.reason == 'timeout' and bool(batched.timeout.all())
    task.reset(); env.reset(full)
    for _ in range(50):
        native = task.evaluate(np.r_[task.target,body.center[2]], np.zeros(3), 0.)
        center = torch.cat((env.target,env.center[:,2:3]),dim=1)
        batched = env.evaluate(center,torch.zeros(2,3,device='cuda'),torch.zeros(2,device='cuda'))
    assert native.reason=='complete' and task.completed==1 and bool(batched.complete.all())
    assert bool((env.completed==1).all())
    write_json(args.output/'checks.json',dict(conditional_density=True,local_identity_rejected=rejected,
               exact_legacy_adam_restore=legacy['exact_adam_next_update'],reset_and_partial_reset=True,
               five_second_timeout=True,one_goal_completion=True,max_short_step_qpos_error=max_qpos))
    print(json.dumps(dict(passed=True,max_short_step_qpos_error=max_qpos)))


if __name__=='__main__':
    main()
