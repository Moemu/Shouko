"""CPU regression checks for independent actor and critic input coordinates."""
import io

import torch

from .full_brain import ConnectomePolicy
from .ppo_yumi import Value
from .test_policy_contract import tiny_graph
from .value_observation import ValueObservationStats, set_new_actor_scales


def main():
    torch.set_num_threads(2)
    torch.manual_seed(52)
    with tiny_graph():
        policy = ConnectomePolicy(observation_size=50, device='cpu')
        with torch.no_grad():
            policy.encoder.weight[:, 47:] = 0
        observation = torch.randn(32, 50)
        value = Value(50)
        optimizer = torch.optim.Adam(value.parameters(), lr=1e-3)
        value(observation).square().mean().backward()
        optimizer.step()
        legacy = dict(value=value.state_dict(), value_optimizer=optimizer.state_dict())
        stats = ValueObservationStats.restore(legacy, policy.obs_mean, policy.obs_std)
        with torch.no_grad():
            before_actor = policy(observation)[0]
            before_value = value(stats.normalize(observation))
            raw_input = stats.normalize(observation).clone()
        set_new_actor_scales(policy, [0.081173, 0.0643464, 0.05])
        with torch.no_grad():
            torch.testing.assert_close(policy(observation)[0], before_actor, rtol=0, atol=0)
            torch.testing.assert_close(value(stats.normalize(observation)), before_value, rtol=0, atol=0)
        assert not torch.equal(stats.std, policy.obs_std)
        buffer = io.BytesIO()
        torch.save(dict(legacy, value_observation_stats=stats.state_dict()), buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        restored = ValueObservationStats.restore(saved, policy.obs_mean, policy.obs_std)
        torch.testing.assert_close(restored.normalize(observation), raw_input, rtol=0, atol=0)
        resumed = Value(50)
        resumed.load_state_dict(saved['value'])
        resumed_opt = torch.optim.Adam(resumed.parameters())
        resumed_opt.load_state_dict(saved['value_optimizer'])
        # A save/reload preserves the next critic update, including Adam moments.
        for network, opt, inputs in [(value, optimizer, raw_input),
                                     (resumed, resumed_opt, restored.normalize(observation))]:
            opt.zero_grad(set_to_none=True)
            network(inputs).square().mean().backward()
            opt.step()
        for a, b in zip(value.parameters(), resumed.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for invalid in [dict(mean=torch.zeros(47), std=torch.ones(47)),
                        dict(mean=torch.zeros(50), std=torch.zeros(50)),
                        dict(mean=torch.full((50,), float('nan')), std=torch.ones(50)), None]:
            try:
                ValueObservationStats.restore(dict(value_observation_stats=invalid), policy.obs_mean, policy.obs_std)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid saved critic coordinates accepted')
        for scales in ([0.1, 0, 0.1], [float('nan'), 1, 1]):
            try:
                set_new_actor_scales(policy, scales)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid actor scales accepted')
        with torch.no_grad():
            policy.encoder.weight[0, 47] = 1e-4
        try:
            set_new_actor_scales(policy, [0.1, 0.1, 0.1])
        except ValueError:
            pass
        else:
            raise AssertionError('Learned actor columns were silently rescaled')
    print('PASS: legacy restore, actor/critic isolation, saved statistics and exact next Adam update')


if __name__ == '__main__':
    main()
