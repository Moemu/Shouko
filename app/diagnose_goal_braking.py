"""Read-only endpoint and replay-fork braking diagnostics; never updates weights."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from .braking_task import FiniteTaskContract, FiniteLocalGoalTask, FiniteLocalGoalEnv, save_fork, restore_fork
from .diagnose_goal_d0 import write_json, measured_step
from .evaluate_goal import file_sha256
from .free_body import FreeBody, ROOT
from .goal_checkpoint import goal_contract, read_goal_payload
from .local_goal import LocalGoalSetup, load_local_policy, action_distribution
from .ppo_yumi import Value
from .value_diagnostics import gae_targets

SOURCES=('app/braking_task.py','app/diagnose_goal_braking.py','app/local_goal.py','app/goal_checkpoint.py',
         'app/free_body.py','app/goal_task.py','app/gpu_goal.py','app/mlp_policy.py','app/ppo_yumi.py',
         'app/value_diagnostics.py','app/diagnose_goal_d0.py')
MODES=('policy','zero','hold_angles','damping','policy_yplus','policy_yminus')


@torch.inference_mode()
def sample_endpoints(checkpoint: Path, reset: Path, seed: int, output: Path, deadline: float) -> dict:
    output.mkdir(exist_ok=False)
    source_sha=file_sha256(checkpoint)
    policy,context=load_local_policy(checkpoint,reset,'cuda')
    payload=read_goal_payload(checkpoint,goal_contract(FreeBody()),'cuda',local_context=context)
    critic=Value(69).cuda().eval().requires_grad_(False);critic.load_state_dict(payload['value'])
    assert all(bool(torch.isfinite(t).all()) for t in critic.state_dict().values())
    assert bool(torch.isfinite(payload['log_std']).all())
    log_std=payload['log_std'].clone();source_cost=payload['settings'].get('training_work_cost',.001)
    before={k:v.clone() for k,v in policy.state_dict().items()}
    env=FiniteLocalGoalEnv(16,210000+seed,LocalGoalSetup(reset,context.rho))
    torch.manual_seed(220000+seed)
    observation=env.observation()
    names=('observation','next_observation','after_reset_observation','action','executed_action','mean','log_prob',
           'value','next_value','after_reset_value','training_reward','common_reward','progress','work_j','reached',
           'timeout','arena','terminated','truncated','done','qpos','qvel','next_qpos','next_qvel','penetration','limit_error')
    records:dict[str,list[np.ndarray]]={name:[] for name in names}
    bounds=torch.as_tensor(env.body.ranges,dtype=torch.float32,device='cuda')
    reason=''
    for _ in range(320):
        if time.time()>=deadline:reason='budget';break
        mean,_=policy(observation);distribution=action_distribution(mean,log_std,observation,context.rho)
        action=distribution.sample();value=critic(observation)
        qpos,qvel=env.qpos.clone(),env.qvel.clone()
        next_observation,result=env.step(action)
        next_value=critic(next_observation)
        q=env.qpos[:,env.qadr]
        error=torch.maximum(bounds[:,0]-q,q-bounds[:,1]).clamp_min(0).amax(1)
        done=result.terminated|result.truncated
        training_reward=result.reward+(.001-source_cost)*result.work_j
        tensors=dict(observation=observation,next_observation=next_observation,action=action,
                     executed_action=env.actions.clone(),mean=mean,log_prob=distribution.log_prob(action).sum(-1),
                     value=value,next_value=next_value,training_reward=training_reward,common_reward=result.reward,
                     progress=result.progress,work_j=result.work_j,reached=result.reached,timeout=result.timeout,
                     arena=result.arena,terminated=result.terminated,truncated=result.truncated,done=done,
                     qpos=qpos,qvel=qvel,next_qpos=env.qpos.clone(),next_qvel=env.qvel.clone(),
                     penetration=env.peak_penetration.clone(),limit_error=error)
        for name,tensor in tensors.items():records[name].append(tensor.cpu().numpy().copy())
        if bool(result.numerical.any()) or not bool(torch.isfinite(next_observation).all()):reason='numerical'
        elif float(env.peak_penetration.max())>.030 or float(error.max())>.15:reason='physical_guard'
        after=next_observation if reason else env.reset(done) if bool(done.any()) else next_observation
        records['after_reset_observation'].append(after.cpu().numpy().copy())
        records['after_reset_value'].append(critic(after).cpu().numpy().copy())
        observation=after
        if reason:break
    arrays={k:np.asarray(v) for k,v in records.items()};count=len(arrays['value'])
    if count:
        old_rewards=arrays['training_reward']+.99*arrays['next_value']*(arrays['timeout']|arrays['arena'])
        new_rewards=arrays['training_reward']+.99*arrays['next_value']*arrays['arena']
        advantages_old=[];advantages_finite=[]
        for start in range(0,count,64):
            stop=min(start+64,count)
            values=torch.from_numpy(np.concatenate((arrays['value'][start:stop],arrays['after_reset_value'][stop-1:stop])))
            done=torch.from_numpy(arrays['done'][start:stop].astype(np.float32))
            for destination,reward in ((advantages_old,old_rewards),(advantages_finite,new_rewards)):
                adv,_=gae_targets(torch.from_numpy(reward[start:stop]),values,done,.99,.95)
                destination.append(adv.numpy())
        arrays.update(legacy_bootstrap_reward=old_rewards,finite_bootstrap_reward=new_rewards,
                      legacy_advantage=np.concatenate(advantages_old),finite_advantage=np.concatenate(advantages_finite))
    arrays.update(log_std=log_std.cpu().numpy())
    np.savez_compressed(output/'trace.npz',**arrays)
    assert source_sha==file_sha256(checkpoint)
    assert all(torch.equal(t,before[k]) for k,t in policy.state_dict().items())
    report=dict(complete=count==320 and not reason,reason=reason,steps=count,transitions=count*16,
                timeout_events=int(arrays['timeout'].sum()),done_events=int(arrays['done'].sum()),
                checkpoint_sha256=source_sha,reset_sha256=file_sha256(reset),trace_sha256=file_sha256(output/'trace.npz'),
                source_context=asdict(context),diagnostic_contract=asdict(FiniteTaskContract()),
                source_training_work_cost=source_cost,weights_unchanged=True)
    if count:
        difference=arrays['legacy_advantage']-arrays['finite_advantage']
        report.update(max_advantage_change=float(np.abs(difference).max()),
                      mean_absolute_advantage_change=float(np.abs(difference).mean()),
                      max_penetration_m=float(arrays['penetration'].max()),max_limit_error_rad=float(arrays['limit_error'].max()))
    write_json(output/'summary.json',report)
    print(json.dumps(dict(endpoint=output.name,transitions=report['transitions'],timeouts=report['timeout_events'],reason=reason)),flush=True)
    return report


def replay_forks(source: Path, output: Path, deadline: float) -> tuple[list[dict],dict]:
    output.mkdir(exist_ok=False)
    rows=[];max_qpos=max_qvel=max_com=max_action=0.
    for seed in (1401,1402,1403):
        with np.load(source/f'eval_iid_zero_{seed}_native/trace.npz') as z:
            arrays={key:z[key].copy() for key in ('qpos','qvel','center','raw_action','observation')}
        policy,context=load_local_policy(source/f'iid_zero_{seed}/last.pt',source/'reset.npz')
        task=FiniteLocalGoalTask(FreeBody(),source/'reset.npz')
        for step in range(251):
            if time.time()>=deadline:raise TimeoutError('Fork replay wall budget')
            max_qpos=max(max_qpos,float(np.abs(task.body.data.qpos-arrays['qpos'][step]).max()))
            max_qvel=max(max_qvel,float(np.abs(task.body.data.qvel-arrays['qvel'][step]).max()))
            max_com=max(max_com,float(np.abs(task.body.center-arrays['center'][step]).max()))
            if step in (100,125,145):
                path=output/f'fork_{seed}_{step}.npz';save_fork(task,path)
                rows.append(dict(id=f'{seed}_{step}',seed=seed,step=step,time_s=step*.02,
                                 filename=path.name,sha256=file_sha256(path),source_checkpoint_sha256=file_sha256(source/f'iid_zero_{seed}/last.pt')))
            if step==250:break
            with torch.inference_mode():
                obs=torch.from_numpy(task.observation())[None]
                mu,_=policy(obs);raw=action_distribution(mu,torch.zeros(18),obs,context.rho).mean[0].numpy()
            max_action=max(max_action,float(np.abs(raw-arrays['raw_action'][step]).max()))
            task.step(arrays['raw_action'][step])
    if max(max_qpos,max_qvel,max_com)>1e-9:
        raise RuntimeError('Saved-action replay differs from archived native trajectory')
    errors=dict(max_qpos_error=max_qpos,max_qvel_error=max_qvel,max_com_error_m=max_com,
                max_policy_action_error=max_action,action_floor=max(1e-6,10*max_action),position_floor_m=max(1e-4,10*max_com))
    write_json(output/'manifest.json',dict(forks=rows,replay=errors))
    return rows,errors


@torch.inference_mode()
def probe(policy: torch.nn.Module, rho: float, fork: Path, mode: str, output: Path, deadline: float) -> dict:
    task=FiniteLocalGoalTask(FreeBody(),fork)
    shift=.2 if mode=='policy_yplus' else -.2 if mode=='policy_yminus' else 0.
    obs=restore_fork(task,fork,shift);body=task.body
    fixed=(body.data.qpos[body.qadr]-body.home)/body.action_scale
    start=float(task.target_elapsed)
    frames=dict(time=[start],observation=[obs.copy()],center=[body.center],velocity=[body.velocity],
                qpos=[body.data.qpos.copy()],qvel=[body.data.qvel.copy()])
    steps={k:[] for k in ('actor_mean','actor_conditional','command','executed_action','joint_target','target_clipped',
                         'torque_last_nm','saturated_substeps','substeps','work_j','reward','progress','penetration',
                         'limit_error','eligible','dwell_seconds','ground_contact')}
    body_names=[body.model.body(i).name for i in range(body.model.nbody)]
    reason='';success=False;max_dwell=task.dwell
    for _ in range(round((5.-start)/.02)):
        if time.time()>=deadline:reason='budget';break
        tensor=torch.from_numpy(obs)[None]
        mu,_=policy(tensor);conditional=action_distribution(mu,torch.zeros(18),tensor,rho).mean[0].numpy()
        command=(np.zeros(18) if mode=='zero' else fixed if mode=='hold_angles' else
                 (body.data.qpos[body.qadr]-body.home)/body.action_scale if mode=='damping' else conditional)
        before=task.previous_distance
        result=measured_step(body,command)
        now=float(frames['time'][-1]+result['substeps']*.002)
        distance=float(np.linalg.norm(task.target-body.center[:2]));eligible=distance<=.3 and np.linalg.norm(body.velocity[:2])<=.2
        if result['reason']:
            reason=result['reason'];reward=before-distance-.001*result['work_j'];progress=before-distance
            obs=task.observation()
        else:
            evaluated=task.evaluate(body.center,body.velocity,result['work_j'])
            obs=task.observation();reward=evaluated.reward;progress=evaluated.progress
            success=evaluated.reason=='complete';reason=evaluated.reason
        max_dwell=max(max_dwell,task.dwell)
        frames['time'].append(now);frames['observation'].append(obs.copy());frames['center'].append(body.center)
        frames['velocity'].append(body.velocity);frames['qpos'].append(body.data.qpos.copy());frames['qvel'].append(body.data.qvel.copy())
        items=dict(actor_mean=mu[0].numpy().copy(),actor_conditional=conditional.copy(),command=np.asarray(command).copy(),
                   executed_action=body.action.copy(),joint_target=result['target'],target_clipped=result['clipped'],
                   torque_last_nm=body.data.qfrc_actuator[body.vadr].copy(),saturated_substeps=result['saturated'],
                   substeps=result['substeps'],work_j=result['work_j'],reward=reward,progress=progress,
                   penetration=result['max_penetration_m'],limit_error=result['max_limit_error_rad'],eligible=eligible,
                   dwell_seconds=task.dwell,ground_contact=[name in result['ground'] for name in body_names])
        for name,item in items.items():steps[name].append(item)
        if reason:break
    arrays={name:np.asarray(items) for name,items in dict(**frames,**steps).items()}
    arrays['target']=task.target.copy()
    np.savez_compressed(output,**arrays)
    safe=reason in ('timeout','complete')
    return dict(mode=mode,goal_y_shift_m=shift,fork_sha256=file_sha256(fork),trace_sha256=file_sha256(output),
                filename=output.name,safe=safe,success=bool(success and safe),reason=reason,
                start_elapsed_s=start,end_elapsed_s=float(frames['time'][-1]),max_dwell_seconds=float(max_dwell),
                final_distance_m=float(np.linalg.norm(task.target-body.center[:2])),
                final_speed_m_s=float(np.linalg.norm(body.velocity[:2])),final_center_m=body.center.tolist(),
                max_penetration_m=max(steps['penetration'],default=0.),max_limit_error_rad=max(steps['limit_error'],default=0.),
                work_j=float(sum(steps['work_j'])),body_names=body_names)


def run(source: Path, output: Path, max_seconds: float) -> dict:
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    started=time.time();deadline=started+max_seconds
    manifest=dict(protocol='goal_braking_diagnostic_v1',finite_contract=asdict(FiniteTaskContract()),
                  started_unix=started,deadline_unix=deadline,max_seconds=max_seconds,
                  modes=MODES,seeds=[1401,1402,1403],fork_times_s=[2.,2.5,2.9],
                  source_sha256={name:file_sha256(ROOT/name) for name in SOURCES},inputs={})
    for group in ('iid_cost','iid_zero','persistent_cost'):
        for seed in (1401,1402,1403):
            name=f'{group}_{seed}';checkpoint=source/name/'last.pt'
            digest=file_sha256(checkpoint)
            report=json.loads((source/name/'summary.json').read_text(encoding='utf-8'))
            assert digest==report['checkpoint_sha256']
            manifest['inputs'][name]=digest
    manifest['source_reset_sha256']=file_sha256(source/'reset.npz')
    write_json(output/'preregistration.json',manifest)
    for name in SOURCES:
        path=output/'source_snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
    endpoints=[]
    (output/'endpoints').mkdir()
    for group in ('iid_cost','iid_zero','persistent_cost'):
        for seed in (1401,1402,1403):
            if time.time()>=deadline:break
            name=f'{group}_{seed}'
            report=sample_endpoints(source/name/'last.pt',source/'reset.npz',seed,output/'endpoints'/name,deadline)
            report.update(group=group,seed=seed);endpoints.append(report)
            write_json(output/'progress.json',dict(phase='endpoints',endpoints=endpoints))
    forks,errors=replay_forks(source,output/'forks',deadline)
    (output/'probes').mkdir()
    menu=[dict(fork_id=f['id'],mode=m,seed=f['seed']) for m in MODES for f in forks]
    write_json(output/'menu.json',dict(trials=menu,source_forks_sha256=file_sha256(output/'forks/manifest.json')))
    policies={seed:load_local_policy(source/f'iid_zero_{seed}/last.pt',source/'reset.npz') for seed in (1401,1402,1403)}
    rows=[];reference_errors=[]
    for trial in menu:
        if time.time()>=deadline:break
        fork=next(f for f in forks if f['id']==trial['fork_id'])
        policy,context=policies[trial['seed']]
        path=output/'probes'/f"{fork['id']}_{trial['mode']}.npz"
        report=probe(policy,context.rho,output/'forks'/fork['filename'],trial['mode'],path,deadline)
        report.update(trial);rows.append(report)
        if trial['mode']=='policy':
            with np.load(path) as z,np.load(source/f"eval_iid_zero_{trial['seed']}_native/trace.npz") as old:
                reference=old['center'][fork['step']:fork['step']+len(z['center'])]
                error=float(np.abs(reference-z['center']).max());reference_errors.append(error)
                report['zero_intervention_com_error_m']=error
            if len(reference_errors)==9:
                errors['max_zero_intervention_com_error_m']=max(reference_errors)
                errors['position_floor_m']=max(1e-4,10*max(reference_errors))
                write_json(output/'thresholds.json',errors)
                if max(reference_errors)>.003:break
        write_json(output/'progress.json',dict(phase='probes',endpoints=endpoints,probes=rows))
        print(json.dumps(dict(probe=path.stem,success=report['success'],reason=report['reason'],max_dwell=report['max_dwell_seconds'])),flush=True)
    responses=[]
    by_id={(r['fork_id'],r['mode']):r for r in rows}
    for fork in forks:
        reference=by_id.get((fork['id'],'policy'))
        if not reference:continue
        for mode in ('policy_yplus','policy_yminus'):
            altered=by_id.get((fork['id'],mode))
            if not altered:continue
            with np.load(output/'probes'/reference['filename']) as a,np.load(output/'probes'/altered['filename']) as b:
                n=min(5,len(a['command']),len(b['command']))
                responses.append(dict(fork_id=fork['id'],mode=mode,
                    actor_rms=float(np.sqrt(np.mean((a['actor_mean'][:n]-b['actor_mean'][:n])**2))),
                    raw_rms=float(np.sqrt(np.mean((a['actor_conditional'][:n]-b['actor_conditional'][:n])**2))),
                    executed_rms=float(np.sqrt(np.mean((a['executed_action'][:n]-b['executed_action'][:n])**2))),
                    final_position_delta_m=float(np.linalg.norm(a['center'][-1]-b['center'][-1])),
                    final_y_delta_m=float(abs(a['center'][-1,1]-b['center'][-1,1]))))
    write_json(output/'responses.json',dict(thresholds=errors,responses=responses))
    positive=sorted({r['fork_id'] for r in rows if r['mode'] in ('zero','hold_angles','damping') and r['success']})
    absent=(len(responses)==18 and all(x['actor_rms']<=errors['action_floor'] and x['executed_rms']<=errors['action_floor']
            and x['final_position_delta_m']<=errors['position_floor_m'] for x in responses))
    clipped=(len(responses)==18 and all(x['actor_rms']>=.02 and x['raw_rms']>=.02 and x['executed_rms']<=errors['action_floor'] for x in responses))
    target_only=(any(r['success'] and r['goal_y_shift_m'] for r in rows)
                 and not any(r['success'] and not r['goal_y_shift_m'] for r in rows))
    branch=('budget_stop' if time.time()>=deadline or len(rows)<54 and len(reference_errors)!=9 else
            'engineering_block' if len(reference_errors)!=9 or max(reference_errors)>.003 else
            'target_or_execution_uncertainty' if absent or clipped or target_only else
            'holding_control_positive' if positive else
            'partial_braking_only' if any(r['max_dwell_seconds']>0 for r in rows if not r['goal_y_shift_m']) else 'route_uncertainty_stop')
    result=dict(branch=branch,endpoint_reports=endpoints,endpoint_transitions=sum(r['transitions'] for r in endpoints),
                endpoint_timeouts=sum(r['timeout_events'] for r in endpoints),probe_count=len(rows),probes=rows,
                positive_forks=positive,explicit_target_response_conflict=absent or clipped,replay=errors,
                elapsed_seconds=time.time()-started,weight_updates=0,preregistration_sha256=file_sha256(output/'preregistration.json'),
                source_sha256=manifest['source_sha256'])
    write_json(output/'summary.json',result)
    print(json.dumps(dict(branch=branch,probe_count=len(rows),positive_forks=positive)),flush=True)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-screen',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-seconds',type=float,default=3600.)
    args=parser.parse_args()
    if not 0<args.max_seconds<=3600:raise ValueError('Use diagnostic budget in (0,3600] seconds')
    run(args.source_screen,args.output,args.max_seconds)


if __name__=='__main__':main()
