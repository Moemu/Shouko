"""Initial full-graph equivalence check; run from the isolated cloud workspace."""
import json
from pathlib import Path

import torch

from app.full_brain import ConnectomePolicy
from app.ppo_yumi import Value
from app.train_full import file_sha256, validated_checkpoint_state
from app.value_observation import ValueObservationStats, set_new_actor_scales

torch.set_num_threads(8)
torch.manual_seed(2026)
checkpoint = Path('runs/yumi_obs50/best.pt')
state_path = checkpoint.with_name('ppo_state_best.pt')
assert file_sha256(checkpoint) == '458fc465a6fb802a338c218fc4be414fc9153376c5d2f76f116518e6cc5aa475'
assert file_sha256(state_path) == '2450081b857fba0eace4c65fbc2423f93011d24a48f9cd173b8abca18c1d5555'
state = validated_checkpoint_state(checkpoint, state_path, 'cuda')
policy = ConnectomePolicy(observation_size=50).eval()
policy.load(checkpoint)
value = Value(50).cuda().eval()
value.load_state_dict(state['value'])
stats = ValueObservationStats.restore(state, policy.obs_mean, policy.obs_std)
batch = torch.load('rollout.pt', map_location='cpu', weights_only=True)
observations = batch['observations'].reshape(-1, 50)[::257][:64].cuda()
with torch.no_grad():
    original = policy(observations)[0]
    repeat = policy(observations)[0]
    original_value = value(stats.normalize(observations))
    set_new_actor_scales(policy, [0.08117300271987915, 0.06434640288352966, 0.05])
    scaled = policy(observations)[0]
    scaled_value = value(stats.normalize(observations))
torch.testing.assert_close(scaled, original, rtol=1e-5, atol=2e-6)
torch.testing.assert_close(scaled_value, original_value, rtol=0, atol=0)
report = dict(samples=len(observations), checkpoint_sha256=file_sha256(checkpoint),
              state_sha256=file_sha256(state_path),
              actor_max_abs_delta=float((scaled-original).abs().max()),
              actor_repeat_max_abs_delta=float((repeat-original).abs().max()),
              critic_max_abs_delta=float((scaled_value-original_value).abs().max()),
              actor_new_std=policy.obs_std[47:].tolist(),
              critic_new_std=stats.std[47:].tolist(), passed=True)
Path('runs/yumi_actor_scale_ab_20260922/precheck.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
