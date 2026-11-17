"""Independent recomputation of local factorial rollouts and deterministic evaluations."""
import argparse
import json
from pathlib import Path
import math

import numpy as np
import torch

from .evaluate_goal import file_sha256
from .free_body import FreeBody, ROOT
from .goal_checkpoint import LocalGoalContext, HoldingGoalContext, CoverageGoalContext, goal_contract, read_goal_payload
from .mlp_policy import MLPPolicy
from .ppo_yumi import Value


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


@torch.inference_mode()
def verify_rollout(folder: Path, label: str, context: LocalGoalContext | HoldingGoalContext | CoverageGoalContext) -> dict:
    metadata = read(folder/(label+'.json'))
    snapshot = folder/metadata['policy_checkpoint']
    trace = folder/(label+'.npz')
    assert file_sha256(snapshot)==metadata['policy_sha256']
    assert file_sha256(trace)==metadata['trace_sha256']
    payload = read_goal_payload(snapshot,goal_contract(FreeBody()),local_context=context)
    actor=MLPPolicy(69,18,device='cpu').eval(); actor.load_state_dict(payload['policy'])
    critic=Value(69).eval(); critic.load_state_dict(payload['value'])
    with np.load(trace) as z:
        obs=z['observation'];action=z['action'].astype(float)
        mu,_=actor(torch.from_numpy(obs.reshape(-1,69)))
        mu=mu.numpy().reshape(action.shape).astype(float)
        center=context.rho*obs[...,48:66]+(1-context.rho)*mu
        sigma=np.clip(np.exp(z['log_std'].astype(float)),.05,1.5)*math.sqrt(1-context.rho**2)
        density=(-.5*((action-center)/sigma)**2-np.log(sigma)-.5*math.log(2*math.pi)).sum(-1)
        density_error=float(np.max(np.abs(density-z['old_logp'])))
        mean_error=float(np.max(np.abs(mu-z['mean'])))
        assert density_error<1e-4 and mean_error<2e-5
        value=critic(torch.from_numpy(obs.reshape(-1,69))).numpy().reshape(obs.shape[:2]).astype(float)
        critic_error=float(np.max(np.abs(value-z['value'])))
        assert critic_error<2e-5
        settings=payload['settings']; gamma=settings['gamma'];lam=settings['lam']
        cost=settings.get('training_work_cost',.001)
        raw=z['progress'].astype(float)+z['reached'].astype(float)-cost*z['work'].astype(float)
        if isinstance(context, HoldingGoalContext):
            potential=context.potential_scale*(gamma*z['next_dwell_fraction'].astype(float)*~z['terminated']-obs[...,67])
            assert np.max(np.abs(potential-z['potential_reward']))<1e-7
            assert not np.any(z['terminated'] & z['truncated'])
            raw+=potential
        reward_error=float(np.max(np.abs(raw-z['raw_reward'])))
        assert reward_error<1e-6
        reward=raw+gamma*z['next_value'].astype(float)*z['truncated']
        assert np.max(np.abs(reward-z['reward']))<1e-6
        alive=1-z['done'].astype(float)
        predicted=z['value'].astype(float)
        future=np.concatenate((predicted[1:],z['bootstrap_value'][None].astype(float)))
        delta=reward+gamma*future*alive-predicted
        advantage=np.zeros_like(delta);carry=np.zeros(delta.shape[1])
        for t in range(len(delta)-1,-1,-1):
            carry=delta[t]+gamma*lam*alive[t]*carry
            advantage[t]=carry
        advantage_error=float(np.max(np.abs(advantage-z['advantage'])))
        assert advantage_error<2e-5
        assert np.max(np.abs(z['returns']-(z['advantage']+z['value'])))<1e-6
    return dict(log_probability_error=density_error,mean_prediction_error=mean_error,
                critic_prediction_error=critic_error,reward_identity_error=reward_error,advantage_error=advantage_error)


@torch.inference_mode()
def verify_evaluation(folder: Path, state_path: Path, checkpoint: Path) -> dict:
    report=read(folder/'summary.json')
    assert report['checkpoint_sha256']==file_sha256(checkpoint)
    assert report['state_sha256']==file_sha256(state_path)
    assert report['trace_sha256']==file_sha256(folder/'trace.npz')
    for name,digest in report['source_sha256'].items():assert file_sha256(ROOT/name)==digest,name
    context=LocalGoalContext(**report['local_context'])
    payload=read_goal_payload(checkpoint,goal_contract(FreeBody()),local_context=context)
    actor=MLPPolicy(69,18,device='cpu').eval();actor.load_state_dict(payload['policy'])
    body=FreeBody()
    max_com_error=0.
    with np.load(folder/'trace.npz') as z,np.load(state_path) as reset:
        np.testing.assert_allclose(z['qpos'][0],reset['qpos'],atol=1e-7,rtol=0)
        np.testing.assert_allclose(z['qvel'][0],reset['qvel'],atol=1e-7,rtol=0)
        assert len(z['time'])==len(z['work_j'])+1
        np.testing.assert_allclose(z['target']-z['origin'],[2.,0.],atol=2e-6,rtol=0)
        for q,v,c in zip(z['qpos'],z['qvel'],z['center']):
            body.data.qpos[:]=q;body.data.qvel[:]=v;body.refresh()
            max_com_error=max(max_com_error,float(np.max(np.abs(body.center-c))))
        assert max_com_error<2e-6
        distances=np.linalg.norm(z['center'][:,:2]-z['target'],axis=1)
        progress=np.diff(-distances)
        assert np.max(np.abs(progress-z['progress']))<2e-6
        assert abs(float(progress.sum())-report['net_progress_m'])<2e-6
        for seconds,key in ((1.,'after_1s_progress_m'),(3.,'after_3s_progress_m')):
            index=min(round(seconds/.02),len(distances)-1)
            assert abs(float(distances[index]-distances[-1])-report[key])<2e-6
        assert abs(float(z['work_j'].sum())-report['work_j'])<1e-5
        bonus=np.zeros(len(progress));bonus[-1]=1 if report['reached'] else 0
        assert np.max(np.abs(progress+bonus-.001*z['work_j']-z['reward']))<2e-6
        mu,_=actor(torch.from_numpy(z['observation'][:-1]))
        expected=context.rho*z['observation'][:-1,48:66]+(1-context.rho)*mu.numpy()
        action_error=float(np.max(np.abs(expected-z['raw_action'])))
        assert action_error<2e-5
        np.testing.assert_allclose(z['executed_action'],np.clip(z['raw_action'],-8,8),atol=1e-7,rtol=0)
        np.testing.assert_allclose(z['observation'][1:,48:66],z['executed_action'],atol=1e-7,rtol=0)
    return dict(max_com_error_m=max_com_error,max_execution_mean_error=action_error)


def verify(output: Path) -> dict:
    torch.set_num_threads(1)
    prereg=read(output/'preregistration.json');report=read(output/'summary.json')
    archived_sources=set()
    def verify_source(name: str, digest: str) -> None:
        if file_sha256(ROOT/name)==digest:
            return
        snapshot=output/'source_snapshot'/name
        assert snapshot.is_file() and file_sha256(snapshot)==digest,name
        archived_sources.add(name)
    assert file_sha256(output/'preregistration.json')==report['preregistration_sha256']
    assert file_sha256(output/'reset.npz')==prereg['state_sha256']==report['state_sha256']
    for name,digest in prereg['source_sha256'].items():verify_source(name,digest)
    initial_by_digest={};paired={};audited=[];transitions=0;partial=[]
    for row in report['runs']:
        folder=output/f"{row['group']}_{row['seed']}"
        if not row['complete']:
            status=read(folder/'training.json')
            context=LocalGoalContext(**status['local_context'])
            assert context.reset_sha256==prereg['state_sha256'] and context.rho==row['rho']
            initial=read_goal_payload(folder/'initial.pt',goal_contract(FreeBody()),local_context=context)
            if row['seed'] not in paired:paired[row['seed']]=initial
            for section in ('policy','value'):
                assert all(torch.equal(t,initial[section][name]) for name,t in paired[row['seed']][section].items())
            assert torch.equal(paired[row['seed']]['log_std'],initial['log_std'])
            count=sum(h['transitions'] for h in status['history'])
            assert count==row.get('actual_transitions',count)==status['env_steps']
            if status.get('physical_guard'):
                assert status['physical_guard']['penetration_m']>.030 or status['physical_guard']['limit_error_rad']>.15
                interrupted=read_goal_payload(folder/'interrupted.pt',goal_contract(FreeBody()),local_context=context)
                assert interrupted['env_steps']==count
                assert count<status['attempted_local_transitions']<=count+1024
                assert any(not torch.equal(t,interrupted['policy'][name]) for name,t in initial['policy'].items())
                partial.append(dict(group=row['group'],seed=row['seed'],completed_update_transitions=count,
                                    attempted_transitions=status['attempted_local_transitions'],physical_guard=status['physical_guard'],
                                    first_rollout_check=verify_rollout(folder,'rollout_1',context),
                                    status_sha256=file_sha256(folder/'training.json'),
                                    interrupted_checkpoint_sha256=file_sha256(folder/'interrupted.pt')))
            else:
                partial.append(dict(group=row['group'],seed=row['seed'],completed_update_transitions=count,
                                    attempted_transitions=status.get('attempted_local_transitions',count),
                                    reason=status.get('error',row.get('error')),
                                    status_sha256=file_sha256(folder/'training.json')))
            continue
        train=read(folder/'summary.json')
        assert train['launch_transitions']==262144 and len(train['history'])==256
        assert sum(h['transitions'] for h in train['history'])==262144
        for name,digest in train['source_sha256'].items():verify_source(name,digest)
        assert file_sha256(folder/'last.pt')==train['checkpoint_sha256']==row['checkpoint_sha256']
        context=LocalGoalContext(**train['local_context'])
        assert context.rho==row['rho'] and context.reset_sha256==prereg['state_sha256']
        initial=read_goal_payload(folder/'initial.pt',goal_contract(FreeBody()),local_context=context)
        final=read_goal_payload(folder/'last.pt',goal_contract(FreeBody()),local_context=context)
        digest=file_sha256(folder/'initial.pt');initial_by_digest[digest]=folder/'initial.pt'
        if row['seed'] not in paired:paired[row['seed']]=initial
        for section in ('policy','value'):
            assert all(torch.equal(t,initial[section][name]) for name,t in paired[row['seed']][section].items())
        assert torch.equal(paired[row['seed']]['log_std'],initial['log_std'])
        assert any(not torch.equal(initial['policy'][name],t) for name,t in final['policy'].items())
        for h in train['history']:
            expected=(h['progress_sum_m']+h['arrivals']-row['work_cost']*h['absolute_work_j'])/h['transitions']
            assert abs(expected-h['reward_mean'])<1e-6
            assert h['max_penetration_m']<=.030 and h['max_joint_limit_error_rad']<=.15
        rollout_checks={label:verify_rollout(folder,label,context) for label in ('rollout_1','rollout_256')}
        audited.append(dict(group=row['group'],seed=row['seed'],rollouts=rollout_checks))
        transitions+=262144
    evaluation_checks=[]
    evaluation_folders=(sorted(output.glob('eval_*'))+sorted(output.glob('initial_*'))
                        +sorted(output.glob('diagnostic_interrupted_*')))
    for folder in evaluation_folders:
        if not folder.is_dir():continue
        evaluation=read(folder/'summary.json');digest=evaluation['checkpoint_sha256']
        if folder.name.startswith('diagnostic_interrupted_'):
            matches=[output/f"{r['group']}_{r['seed']}"/'interrupted.pt' for r in report['runs'] if not r['complete']]
            checkpoint=next(p for p in matches if p.exists() and file_sha256(p)==digest)
        elif folder.name.startswith('initial_'):
            checkpoint=initial_by_digest[digest]
        else:
            row=next(r for r in report['runs'] if r.get('checkpoint_sha256')==digest)
            checkpoint=output/f"{row['group']}_{row['seed']}"/'last.pt'
        evaluation_checks.append(dict(folder=folder.name,**verify_evaluation(folder,output/'reset.npz',checkpoint)))
    counts={group['group']:0 for group in report['groups']}
    for row in report['runs']:
        if not row['complete']:continue
        passed=True
        for backend in ('native','cuda'):
            f,i=row['final'][backend],row['initial'][backend]
            result=bool(f['safe'] and f['net_progress_m']>=.25 and f['net_progress_m']-i['net_progress_m']>=.15
                        and (f['reached'] or (f['after_1s_progress_m']>=.20 and f['after_3s_progress_m']>=.05)))
            assert result==row['backend_passed'][backend]
            passed &= result
        assert passed==row['passed'];counts[row['group']]+=int(passed)
    for group in report['groups']:assert group['passed_seeds']==counts[group['group']]
    assert transitions==sum(r.get('actual_transitions',0) for r in report['runs'] if r['complete'])
    assert report['actual_transitions']==transitions+sum(r['completed_update_transitions'] for r in partial)
    if report['full_factorial']:assert len(audited)==12 and transitions==3145728
    contrasts=[]
    if report['full_factorial']:
        for seed in (1401,1402,1403):
            rows={r['group']:r for r in report['runs'] if r['seed']==seed}
            values={k:v['final']['native']['net_progress_m'] for k,v in rows.items()}
            contrasts.append(dict(seed=seed,
                persistence_with_cost_m=values['persistent_cost']-values['iid_cost'],
                persistence_without_cost_m=values['persistent_zero']-values['iid_zero'],
                remove_cost_iid_m=values['iid_zero']-values['iid_cost'],
                remove_cost_persistent_m=values['persistent_zero']-values['persistent_cost'],
                interaction_m=(values['persistent_zero']-values['iid_zero'])-(values['persistent_cost']-values['iid_cost'])))
    pairwise=[]
    for label,left,right in (('remove_cost_iid','iid_cost','iid_zero'),
                              ('persistence_with_cost','iid_cost','persistent_cost'),
                              ('remove_cost_persistent','persistent_cost','persistent_zero'),
                              ('persistence_without_cost','iid_zero','persistent_zero')):
        differences=[]
        for seed in (1401,1402,1403):
            a=next((r for r in report['runs'] if r['group']==left and r['seed']==seed and r['complete']),None)
            b=next((r for r in report['runs'] if r['group']==right and r['seed']==seed and r['complete']),None)
            if a and b:
                differences.append(dict(seed=seed,
                    net_progress_delta_m=b['final']['native']['net_progress_m']-a['final']['native']['net_progress_m'],
                    after_3s_delta_m=b['final']['native']['after_3s_progress_m']-a['final']['native']['after_3s_progress_m']))
        pairwise.append(dict(contrast=label,complete_pairs=len(differences),
                             differences=differences if len(differences)==3 else [],
                             limitation='Three paired training seeds; no interaction estimate from incomplete cells'))
    result=dict(verified=True,full_factorial=report['full_factorial'],verified_transitions=transitions,
                archived_sources_used=sorted(archived_sources),
                total_attempted_training_transitions=transitions+sum(r['attempted_transitions'] for r in partial),
                passed_seeds=counts,rollout_checks=audited,partial_runs=partial,evaluation_checks=evaluation_checks,
                paired_contrasts=contrasts,verifier_sha256=file_sha256(Path(__file__)),
                complete_cell_pairwise=pairwise,
                campaign_summary_sha256=file_sha256(output/'summary.json'))
    (output/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(verified=True,verified_transitions=transitions,passed_seeds=counts)))
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();verify(args.output)


if __name__=='__main__':main()
