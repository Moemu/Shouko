"""Matched readout training with an optional single world-vy sensory column."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from .evaluate_locomotion import load_policy
from .imitate_yumi import dataset, metrics, predict
from .ordered_inference import ordered_inference
from .train_full import atomic_json, file_sha256


class VyEncoder(nn.Module):
    """Keep the original matrix multiply, with only column 48 trainable."""
    def __init__(self, encoder):
        super().__init__()
        if encoder.in_features != 50 or encoder.bias is not None:
            raise ValueError('Expected the bias-free 50-input encoder')
        self.in_features, self.out_features = encoder.in_features, encoder.out_features
        self.register_buffer('fixed_weight', encoder.weight.detach().clone())
        self.register_buffer('column_index', torch.tensor([48], device=encoder.weight.device))
        self.column = nn.Parameter(encoder.weight[:, 48:49].detach().clone())

    @property
    def weight(self):
        return self.fixed_weight.index_copy(1, self.column_index, self.column)

    def forward(self, observation):
        return F.linear(observation, self.weight)

    @torch.no_grad()
    def fold(self, encoder):
        encoder.weight.copy_(self.weight)
        return encoder


@torch.no_grad()
def input_response(model, observations, velocity_scale):
    result = {}
    with ordered_inference():
        base = model(observations)[0]
        for sign in [-1, 1]:
            shifted = observations.clone()
            shifted[:, 48] += sign * .05 * velocity_scale
            result[str(sign)] = float((model(shifted)[0]-base).abs().max())
    return result


def check_boundary(before, after, sensory):
    changes = {}
    for name, value in after.items():
        old = before[name].to(value.device)
        if torch.equal(old, value):
            continue
        difference = (value-old).abs()
        changes[name] = float(difference.max())
        if name.startswith('readout.'):
            continue
        if name == 'encoder.weight' and sensory:
            difference[:, 48] = 0
        elif name == 'obs_std':
            difference[48] = 0
        else:
            raise AssertionError(f'Frozen tensor changed: {name}')
        if torch.count_nonzero(difference):
            raise AssertionError(f'Frozen values changed: {name}')
    return changes


def grouped_metrics(prediction, data):
    observations = data['observations']
    speed = observations[:, 6]/data['physics_interface']['cmd_scale'][0]
    phase = torch.remainder(torch.atan2(observations[:, 45], observations[:, 46])+torch.pi, 2*torch.pi)
    vy = observations[:, 48]/data['physics_interface']['linear_velocity_scale']
    masks = {f'speed_{v}': (speed-v).abs() < 1e-5 for v in [.35, .5, .65]}
    masks.update({f'phase_quarter_{i}': (phase >= i*torch.pi/2) &
                  (phase < (i+1)*torch.pi/2) for i in range(4)})
    masks.update(vy_negative=vy < -.02, vy_central=vy.abs() <= .02, vy_positive=vy > .02)
    return dict(overall=metrics(prediction, data['targets']),
                groups={name: dict(samples=int(mask.sum()), mse=float((prediction[mask]-data['targets'][mask]).double().square().mean()))
                        for name, mask in masks.items() if mask.any()})


def train(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    model, cfg, kind = load_policy(args.source)
    if kind != 'connectome':
        raise ValueError('Expected a full-connectome source')
    source = torch.load(args.source, map_location='cpu', weights_only=True, mmap=True)
    before = source['state_dict']
    training = dataset(args.train)
    validation = {p: dataset([p]) for p in args.validation}
    train_ids = set(training['episode_ids'].tolist())
    for data in validation.values():
        if train_ids & set(data['episode_ids'].tolist()):
            raise ValueError('Training and validation episodes overlap')
        if any(data[k] != training[k] for k in ['teacher_sha256', 'physics_interface']):
            raise ValueError('Validation provenance mismatch')
    model.load(args.source, interface=training['physics_interface'])
    probe = training['observations'][:32].cuda()
    with ordered_inference():
        original_prediction = model(probe)[0]
    if args.calibrate_vy:
        if torch.count_nonzero(model.encoder.weight[:, 48]):
            raise ValueError('Initial scale calibration requires a zero vy column')
        model.obs_std[48] = training['observations'][:, 48].std().clamp_min(1e-4)
    model.requires_grad_(False)
    model.readout.requires_grad_(True)
    original_encoder = model.encoder
    if args.sensory:
        model.encoder = VyEncoder(original_encoder)
    with ordered_inference():
        if not torch.equal(original_prediction, model(probe)[0]):
            raise AssertionError('Initialization changed actions')
    groups = [dict(params=list(model.readout.parameters()), lr=args.readout_lr)]
    if args.sensory:
        groups.append(dict(params=[model.encoder.column], lr=args.sensory_lr))
    optimizer = torch.optim.Adam(groups, foreach=False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    report = dict(completed=False, arguments=vars(args), source_checkpoint_sha256=file_sha256(args.source),
                  source_sha256={name: file_sha256(Path(__file__).parent/name) for name in
                                 ['train_sensory_readout.py', 'full_brain.py', 'ordered_inference.py']},
                  data_sha256={p: file_sha256(p) for p in args.train+args.validation},
                  trainable_parameters=sum(p.numel() for p in parameters), initialization_identical=True,
                  vy_observation_std=float(model.obs_std[48]), history=[])
    started = time.monotonic()
    def save():
        report['wall_seconds'] = time.monotonic()-started
        atomic_json(out/'fit.json', report)
    def validate():
        return {p: grouped_metrics(predict(model, data['observations'], args.batch), data)
                for p, data in validation.items()}
    report['initial_input_response'] = input_response(model, probe, training['physics_interface']['linear_velocity_scale'])
    report['initial_validation'] = validate()
    save()
    tx, ty = training['observations'].cuda(), training['targets'].cuda()
    generator = torch.Generator().manual_seed(args.seed)
    index_digest = hashlib.sha256()
    for update in range(1, args.updates+1):
        indices = torch.randint(len(tx), (args.batch,), generator=generator)
        index_digest.update(indices.numpy().tobytes())
        indices = indices.cuda()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(tx[indices])[0]
        loss = (prediction-ty[indices]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite loss')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError('Missing or non-finite gradient')
        if update in (1, args.updates):
            report[f'gradient_update_{update}'] = {n: float(p.grad.abs().max())
                for n, p in model.named_parameters() if p.requires_grad}
        optimizer.step()
        if update == 1 or update % 50 == 0 or update == args.updates:
            row = dict(update=update, batch_mse=float(loss), elapsed_seconds=time.monotonic()-started)
            report['history'].append(row)
            save()
            print(json.dumps(row), flush=True)
    report['sample_indices_sha256'] = index_digest.hexdigest()
    report['final_input_response'] = input_response(model, probe, training['physics_interface']['linear_velocity_scale'])
    report['final_validation'] = validate()
    if args.sensory:
        with ordered_inference():
            prediction = model(probe)[0]
            model.encoder = model.encoder.fold(original_encoder)
            if not torch.equal(prediction, model(probe)[0]):
                raise AssertionError('Folding changed actions')
    report['parameter_changes'] = check_boundary(before, model.state_dict(), args.sensory)
    model.save(out/'last.pt', extra=dict(method='vy-column and readout' if args.sensory else 'readout Adam',
               source_sha256=report['source_checkpoint_sha256'], updates=args.updates),
               physics=training['physics_interface'])
    report.update(completed=True, checkpoint_sha256=file_sha256(out/'last.pt'),
                  peak_memory_bytes=torch.cuda.max_memory_allocated())
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--train', nargs='+', required=True)
    parser.add_argument('--validation', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sensory', action='store_true')
    parser.add_argument('--calibrate-vy', action='store_true')
    parser.add_argument('--updates', type=int, default=500)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--seed', type=int, default=3031)
    parser.add_argument('--readout-lr', type=float, default=1e-5)
    parser.add_argument('--sensory-lr', type=float, default=1e-4)
    args = parser.parse_args()
    if args.updates < 1 or args.batch < 1:
        parser.error('Updates and batch must be positive')
    train(args)


if __name__ == '__main__':
    main()
