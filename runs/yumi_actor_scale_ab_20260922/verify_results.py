"""Verify checkpoint pairs and measure fixed-observation feedback after online A/B."""
import json
from pathlib import Path

import torch

from app.full_brain import ConnectomePolicy
from app.train_full import file_sha256, validated_checkpoint_state
from app.value_observation import ValueObservationStats

torch.set_num_threads(8)
root = Path('runs/yumi_actor_scale_ab_20260922')
source = Path('runs/yumi_obs50/best.pt')
source_weights = torch.load(source, map_location='cpu', weights_only=True, mmap=True)['state_dict']
batch = torch.load('rollout.pt', map_location='cpu', weights_only=True)
observations = batch['observations'].reshape(-1, 50)
assert batch['checkpoint_sha256'] == file_sha256(source)
probe = observations[torch.linspace(0, len(observations)-1, 64).long()].cuda()
raw_std = observations.std(0).cuda()
policy = ConnectomePolicy(observation_size=50).eval()
report = dict(kind='online_ab_checkpoint_and_fixed_input_diagnostics',
              observations=64, rollout_sha256=file_sha256(Path('rollout.pt')),
              source_checkpoint_sha256=file_sha256(source), results=[],
              limitation='Fixed source rollout input shifts are sensitivity checks, not gait evidence.')


def rms(value):
    return float(value.double().square().mean().sqrt())


with torch.inference_mode():
    for name, checkpoint in [('source', source), ('control', root/'control/last.pt'),
                             ('scaled', root/'scaled/last.pt')]:
        state_path = checkpoint.with_name('ppo_state_best.pt' if name == 'source' else 'ppo_state.pt')
        state = validated_checkpoint_state(checkpoint, state_path, 'cpu')
        policy.load(checkpoint)
        stats = ValueObservationStats.restore(state, policy.obs_mean.cpu(), policy.obs_std.cpu())
        torch.testing.assert_close(stats.mean, source_weights['obs_mean'], rtol=0, atol=0)
        torch.testing.assert_close(stats.std, source_weights['obs_std'], rtol=0, atol=0)
        base = policy(probe)[0]
        repeat = policy(probe)[0]
        ablated = probe.clone()
        ablated[:, 47:] = 0
        shifts = []
        for column, label in [(47, 'vx'), (48, 'vy'), (49, 'height')]:
            normalized = (probe[:, column] - policy.obs_mean[column]) / policy.obs_std[column]
            changed = probe.clone()
            changed[:, column] += raw_std[column]
            plus = policy(changed)[0]
            changed[:, column] -= 2 * raw_std[column]
            minus = policy(changed)[0]
            shifts.append(dict(column=column, label=label, raw_std=float(raw_std[column]),
                               actor_std=float(policy.obs_std[column]),
                               normalized_min=float(normalized.min()), normalized_max=float(normalized.max()),
                               clamp_fraction=float((normalized.abs() >= 10).float().mean()),
                               plus_one_std_action_rms=rms(plus-base), minus_one_std_action_rms=rms(minus-base)))
        row = dict(name=name, checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                   state_sha256=file_sha256(state_path), critic_coordinates_preserved=True,
                   critic_new_std=stats.std[47:].tolist(),
                   actor_new_std=policy.obs_std[47:].tolist(),
                   new_encoder_rms=rms(policy.encoder.weight[:, 47:]),
                   repeat_action_rms=rms(repeat-base),
                   zero_new_inputs_action_rms=rms(policy(ablated)[0]-base),
                   input_shifts=shifts)
        if name != 'source':
            status = json.loads((checkpoint.parent/'training.json').read_text())
            row['iteration'] = status['iteration']
            row['actor_rounds'] = sum(not h['value_warmup'] for h in status['history'])
            row['optimizer_steps'] = sum(h['policy_optimizer_steps'] for h in status['history'])
            row['saved_adam_steps'] = sorted(set(float(slot['step']) for slot in state['optimizer']['state'].values()))
            assert row['iteration'] == 7 and row['actor_rounds'] == 6
            assert row['saved_adam_steps'] == [float(row['optimizer_steps'])]
        report['results'].append(row)
assert file_sha256(source) == '458fc465a6fb802a338c218fc4be414fc9153376c5d2f76f116518e6cc5aa475'
assert file_sha256(source.with_name('ppo_state_best.pt')) == '2450081b857fba0eace4c65fbc2423f93011d24a48f9cd173b8abca18c1d5555'
report['source_unchanged'] = True
(root/'verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
