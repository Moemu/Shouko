"""Verify selected-column gradients and standard checkpoint folding on a tiny graph."""
import torch

from .full_brain import ConnectomePolicy
from .test_policy_contract import tiny_graph
from .train_sensory_readout import VyEncoder, check_boundary


def main():
    torch.manual_seed(32)
    with tiny_graph():
        model = ConnectomePolicy(observation_size=50, device='cpu')
        with torch.no_grad():
            model.encoder.weight[:, 47:50] = 0
        before = {k: v.clone() for k, v in model.state_dict().items()}
        x = torch.randn(16, 50)
        with torch.no_grad():
            initial = model(x)[0]
        model.obs_std[48] = .003
        model.requires_grad_(False)
        model.readout.requires_grad_(True)
        original = model.encoder
        model.encoder = VyEncoder(original)
        with torch.no_grad():
            assert torch.equal(initial, model(x)[0])
        reference = ConnectomePolicy(observation_size=50, device='cpu')
        reference.load_state_dict(before)
        reference.obs_std[48] = .003
        reference.requires_grad_(False)
        reference.encoder.requires_grad_(True)
        target = torch.randn(16, 12)
        (reference(x)[0]-target).square().mean().backward()
        loss = (model(x)[0]-target).square().mean()
        loss.backward()
        torch.testing.assert_close(model.encoder.column.grad, reference.encoder.weight.grad[:, 48:49])
        assert model.encoder.column.grad.abs().max() > 0
        torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001).step()
        with torch.no_grad():
            trained = model(x)[0]
            model.encoder = model.encoder.fold(original)
            assert torch.equal(trained, model(x)[0])
        changes = check_boundary(before, model.state_dict(), True)
        assert 'encoder.weight' in changes and 'readout.weight' in changes
        # Corrupting a neighbouring sensory column must fail the boundary audit.
        with torch.no_grad():
            model.encoder.weight[0, 47] += .01
        try:
            check_boundary(before, model.state_dict(), True)
        except AssertionError:
            pass
        else:
            raise AssertionError('An unrelated encoder change was accepted')
    print('Selected-column gradient, initialization, folding and boundary checks passed')


if __name__ == '__main__':
    main()
