"""Independent endpoint-mask, GAE, replay-state and holding-event verification."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .evaluate_goal import file_sha256
from .free_body import FreeBody, ROOT
from .goal_checkpoint import goal_contract, LocalGoalContext, read_goal_payload
from .mlp_policy import MLPPolicy
from .ppo_yumi import Value


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def gae(reward: np.ndarray, value: np.ndarray, after_reset: np.ndarray, done: np.ndarray) -> np.ndarray:
    result=np.zeros_like(reward,dtype=float)
    for start in range(0,len(reward),64):
        stop=min(start+64,len(reward));carry=np.zeros(reward.shape[1])
        future=np.concatenate((value[start+1:stop],after_reset[stop-1:stop]))
        delta=reward[start:stop]+.99*future*(1-done[start:stop])-value[start:stop]
        for local in range(stop-start-1,-1,-1):
            carry=delta[local]+.99*.95*(1-done[start+local])*carry;result[start+local]=carry
    return result


@torch.inference_mode()
def verify(output: Path, source: Path) -> dict:
    torch.set_num_threads(1)
    manifest=read(output/'preregistration.json');summary=read(output/'summary.json')
    assert file_sha256(output/'preregistration.json')==summary['preregistration_sha256']
    for name,digest in manifest['source_sha256'].items():
        assert file_sha256(output/'source_snapshot'/name)==digest,name
    assert file_sha256(source/'reset.npz')==manifest['source_reset_sha256']
    endpoint_checks=[];max_density=max_gae=max_reward=max_prediction=0.
    body=FreeBody()
    for report in summary['endpoint_reports']:
        name=f"{report['group']}_{report['seed']}";folder=output/'endpoints'/name
        assert file_sha256(folder/'trace.npz')==report['trace_sha256']
        checkpoint=source/name/'last.pt';assert file_sha256(checkpoint)==report['checkpoint_sha256']==manifest['inputs'][name]
        context=LocalGoalContext(**report['source_context'])
        payload=read_goal_payload(checkpoint,goal_contract(body),local_context=context)
        actor=MLPPolicy(69,18,device='cpu').eval();actor.load_state_dict(payload['policy'])
        critic=Value(69).eval();critic.load_state_dict(payload['value'])
        with np.load(folder/'trace.npz') as z:
            count=len(z['value']);obs=z['observation'];next_obs=z['next_observation']
            if not count:continue
            mu,_=actor(torch.from_numpy(obs.reshape(-1,69)));mu=mu.numpy().reshape(z['action'].shape)
            center=context.rho*obs[...,48:66]+(1-context.rho)*mu
            sigma=np.clip(np.exp(z['log_std'].astype(float)),.05,1.5)*math.sqrt(1-context.rho**2)
            logp=(-.5*((z['action']-center)/sigma)**2-np.log(sigma)-.5*math.log(2*math.pi)).sum(-1)
            max_density=max(max_density,float(np.abs(logp-z['log_prob']).max()))
            for x,y in ((obs,z['value']),(next_obs,z['next_value']),(z['after_reset_observation'],z['after_reset_value'])):
                predicted=critic(torch.from_numpy(x.reshape(-1,69))).numpy().reshape(y.shape)
                max_prediction=max(max_prediction,float(np.abs(predicted-y).max()))
            cost=payload['settings'].get('training_work_cost',.001)
            raw=z['progress'].astype(float)+z['reached']-cost*z['work_j']
            max_reward=max(max_reward,float(np.abs(raw-z['training_reward']).max()))
            assert np.array_equal(z['done'],z['terminated']|z['truncated'])
            assert np.all(z['terminated'][z['timeout']]) and not np.any(z['truncated'][z['timeout']])
            np.testing.assert_allclose(next_obs[...,68][z['timeout']],0.,atol=1e-6)
            if report['reason'] not in ('physical_guard','numerical'):
                np.testing.assert_allclose(z['after_reset_observation'][...,68][z['done']],1.,atol=1e-6)
            old=raw+.99*z['next_value']*(z['timeout']|z['arena'])
            finite=raw+.99*z['next_value']*z['arena']
            for reward,key in ((old,'legacy_advantage'),(finite,'finite_advantage')):
                expected=gae(reward,z['value'].astype(float),z['after_reset_value'].astype(float),z['done'].astype(float))
                max_gae=max(max_gae,float(np.abs(expected-z[key]).max()))
            assert report['timeout_events']==int(z['timeout'].sum())
            endpoint_checks.append(dict(id=name,transitions=count*16,timeouts=int(z['timeout'].sum()),
                                       endpoint_values=z['next_value'][z['timeout']].tolist()))
    assert max_density<1e-4 and max_prediction<5e-5 and max_reward<1e-6 and max_gae<2e-5
    forks=read(output/'forks/manifest.json')['forks'];rows=[];max_com=max_speed=0.;positive=set()
    for fork in forks:
        path=output/'forks'/fork['filename'];assert file_sha256(path)==fork['sha256']
        with np.load(path) as a,np.load(source/f"eval_iid_zero_{fork['seed']}_native/trace.npz") as b:
            np.testing.assert_array_equal(a['qpos'],b['qpos'][fork['step']])
            np.testing.assert_array_equal(a['qvel'],b['qvel'][fork['step']])
            np.testing.assert_array_equal(a['previous_action'],b['executed_action'][fork['step']-1])
    for report in summary['probes']:
        path=output/'probes'/report['filename'];assert file_sha256(path)==report['trace_sha256']
        fork=next(f for f in forks if f['id']==report['fork_id'])
        with np.load(path) as z,np.load(output/'forks'/fork['filename']) as initial:
            np.testing.assert_array_equal(z['qpos'][0],initial['qpos'])
            np.testing.assert_array_equal(z['qvel'][0],initial['qvel'])
            assert abs(z['time'][0]-float(initial['elapsed']))<1e-10 and z['time'][-1]<=5+1e-9
            target=initial['target'].copy();target[1]+=report['goal_y_shift_m']
            np.testing.assert_array_equal(z['target'],target)
            expected=0. if report['goal_y_shift_m'] else float(initial['dwell']);peak=expected
            for i,(q,v,center,speed) in enumerate(zip(z['qpos'],z['qvel'],z['center'],z['velocity'])):
                body.data.qpos[:]=q;body.data.qvel[:]=v;body.refresh()
                max_com=max(max_com,float(np.abs(body.center-center).max()))
                max_speed=max(max_speed,float(np.abs(body.velocity-speed).max()))
                if i:
                    last_aborted=i==len(z['work_j']) and report['reason'] in ('penetration','joint_limit','numerical_or_warning')
                    if not last_aborted:
                        eligible=np.linalg.norm(center[:2]-target)<=.3 and np.linalg.norm(speed[:2])<=.2
                        expected=expected+.02 if eligible else 0.
                    assert abs(expected-z['dwell_seconds'][i-1])<1e-9
                    peak=max(peak,expected)
            np.testing.assert_allclose(z['executed_action'],np.clip(z['command'],-8,8),atol=1e-9,rtol=0)
            commanded=np.clip(body.home+.25*z['executed_action'],body.ranges[:,0],body.ranges[:,1])
            np.testing.assert_allclose(z['joint_target'],commanded,atol=1e-9,rtol=0)
            assert np.all(np.abs(z['torque_last_nm'])<=body.limits+1e-7)
            success=bool(peak>=1-1e-9 and report['safe'] and report['reason']=='complete')
            assert success==report['success'] and abs(peak-report['max_dwell_seconds'])<1e-9
            if success and report['mode'] in ('zero','hold_angles','damping'):positive.add(report['fork_id'])
            rows.append(dict(id=path.stem,success=success,max_dwell_seconds=peak))
    assert max_com<1e-9 and max_speed<1e-9
    assert sorted(positive)==summary['positive_forks']
    result=dict(verified=True,endpoint_transitions=sum(r['transitions'] for r in endpoint_checks),
                timeout_events=sum(r['timeouts'] for r in endpoint_checks),probe_count=len(rows),positive_forks=sorted(positive),
                max_log_probability_error=max_density,max_value_prediction_error=max_prediction,
                max_reward_identity_error=max_reward,max_advantage_error=max_gae,max_com_error_m=max_com,max_velocity_error=max_speed,
                endpoint_checks=endpoint_checks,probe_checks=rows,summary_sha256=file_sha256(output/'summary.json'),
                verifier_sha256=file_sha256(Path(__file__)))
    (output/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('verified','endpoint_transitions','timeout_events','positive_forks')}))
    return result


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-screen',type=Path,required=True)
    a=p.parse_args();verify(a.output,a.source_screen)


if __name__=='__main__':main()
