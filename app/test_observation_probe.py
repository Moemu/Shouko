"""Small CPU checks for the offline observation experiment's controls."""
import math
import torch

from .full_brain import ConnectomePolicy
from .ppo_yumi import Value, freeze_policy_core
from .probe_observation_learning import critic_probe, optimizer_for
from .test_policy_contract import tiny_graph


def main():
    torch.set_num_threads(2)
    torch.manual_seed(37)
    observation = torch.randn(40, 50)
    mean = torch.randn(50)
    old_scale = torch.rand(50) + 0.2
    new_scale = old_scale.clone()
    new_scale[47:] = torch.tensor([0.08, 0.06, 0.05])
    value = Value(50)
    result = critic_probe(dict(value=value.state_dict()), observation, torch.randn(40),
                          mean, old_scale, new_scale)
    assert result['naive_output_rms_change'] > 0.01
    assert result['preserved_max_abs_error'] < 1e-5
    with tiny_graph():
        policy = ConnectomePolicy(observation_size=50, device='cpu')
        freeze_policy_core(policy)
        with torch.no_grad():
            policy.encoder.weight[:, 47:] = 0
            policy.obs_mean.copy_(mean)
            policy.obs_std.copy_(old_scale)
            original, _ = policy(observation)
            policy.obs_std.copy_(new_scale)
            changed, _ = policy(observation)
        torch.testing.assert_close(original, changed, rtol=0, atol=0)
        names = [['neuron_bias'], ['readout.weight', 'readout.bias'],
                 ['encoder.weight'], ['log_std']]
        std = torch.nn.Parameter(torch.zeros(12))
        parameters = dict(policy.named_parameters(), log_std=std)
        old_optimizer = torch.optim.Adam([
            dict(params=[parameters[n] for n in group], lr=0.01) for group in names])
        state = dict(optimizer=old_optimizer.state_dict(), optimizer_names=names)
        optimizer = optimizer_for(policy, std, state, 5e-6)
        assert all(math.isclose(g['lr'], expected) for g, expected in
                   zip(optimizer.param_groups, [5e-7, 5e-6, 2.5e-6, 5e-7]))
        # Adam largely cancels a constant gradient multiplier on the first step.
        parameter = torch.nn.Parameter(torch.zeros(2))
        adam = torch.optim.Adam([parameter], lr=2.5e-6)
        parameter.grad = torch.tensor([0.1, 2.0])
        adam.step()
        torch.testing.assert_close(parameter[0], parameter[1], rtol=1e-5, atol=1e-10)
    print('PASS: critic preservation, zero-column actor equivalence, restored rates, Adam scaling')


if __name__ == '__main__':
    main()
