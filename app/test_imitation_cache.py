"""CPU regressions for readout-independent feature identity and cache integrity."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from . import imitate_yumi
from .full_brain import ConnectomePolicy
from .test_policy_contract import tiny_graph


def main():
    torch.set_num_threads(2)
    with tiny_graph(), TemporaryDirectory() as directory:
        model = ConnectomePolicy(observation_size=50, device='cpu')
        original = imitate_yumi.feature_identity(model, 128)
        with torch.no_grad():
            model.readout.bias.add_(.2)
        assert imitate_yumi.feature_identity(model, 128) == original
        with torch.no_grad():
            model.obs_mean[0] += .1
        changed = imitate_yumi.feature_identity(model, 128)
        assert changed != original
        model.neural_steps += 1
        assert imitate_yumi.feature_identity(model, 128) != changed
        root = Path(directory); data = root/'data.pt'
        torch.save(dict(observations=torch.zeros(3, 50)), data)
        expected = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        with patch.object(imitate_yumi, 'predict', return_value=expected) as predict:
            first, audit = imitate_yumi.cached_features(model, [data], 128, root/'cache', original)
            second, reused = imitate_yumi.cached_features(model, [data], 128, root/'cache', original)
            assert predict.call_count == 1 and torch.equal(first, second)
            assert not audit[0]['reused'] and reused[0]['reused']
        cached = next((root/'cache').glob('*.pt'))
        with cached.open('ab') as handle:
            handle.write(b'corruption')
        try:
            imitate_yumi.cached_features(model, [data], 128, root/'cache', original)
        except ValueError as error:
            assert 'checksum' in str(error)
        else:
            raise AssertionError('Corrupted cache was accepted')
    print('Readout independence, normalization/dynamics invalidation, cache reuse and integrity checks passed')


if __name__ == '__main__':
    main()
