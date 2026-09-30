"""CPU check: cached-feature PPO actions and gradients match the frozen policy."""
import torch

from .full_brain import ConnectomePolicy
from .ppo_yumi import freeze_policy_readout
from .test_policy_contract import tiny_graph


def main():
    torch.set_num_threads(2)
    torch.manual_seed(3029)
    with tiny_graph():
        policy = ConnectomePolicy(observation_size=50, device='cpu')
        freeze_policy_readout(policy)
        assert {name for name, p in policy.named_parameters() if p.requires_grad} == {'readout.weight', 'readout.bias'}
        original = {name: p.clone() for name, p in policy.state_dict().items()}
        observations = torch.randn(16, 50)
        with torch.no_grad():
            expected, activity = policy(observations)
            features = policy.normalizer(activity[policy.outputs].T)
        torch.testing.assert_close(policy.readout(features), expected, rtol=0, atol=0)
        optimizer = torch.optim.SGD(policy.readout.parameters(), lr=.001)
        target = torch.randn(16, 12)
        optimizer.zero_grad()
        (policy(observations)[0]-target).square().mean().backward()
        gradients = {name: p.grad.clone() for name, p in policy.named_parameters() if p.grad is not None}
        optimizer.zero_grad()
        (policy.readout(features)-target).square().mean().backward()
        for name, p in policy.named_parameters():
            if name in gradients:
                torch.testing.assert_close(p.grad, gradients[name], rtol=0, atol=0)
            else:
                assert p.grad is None
        optimizer.step()
        changed = {name for name, p in policy.state_dict().items() if not torch.equal(p, original[name])}
        assert changed == {'readout.weight', 'readout.bias'}
    print('Frozen-core actions, cached-feature gradients and readout-only updates match')


if __name__ == '__main__':
    main()
