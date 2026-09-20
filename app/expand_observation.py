"""Expand a 47-dim-observation checkpoint to the 50-dim interface one-off.

Appends three observation channels (gpu_body.observation, 50-dim branch):
qvel[0:2]*0.25 (body linear velocity) and relative pelvis height. The encoder's
three new input columns are zero-initialized and obs_mean/obs_std are padded
with 0/1, so the expanded policy reproduces the source checkpoint's actions
bit-for-bit at load time (verify with the warm-start check before training).
"""
import argparse
import hashlib
import json

import torch

from .full_brain import ROOT

NEW_DIMS = 3


def file_sha256(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main(args):
    source = ROOT / args.checkpoint
    target = ROOT / args.output
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    state = checkpoint['state_dict']
    old_size = int(checkpoint.get('observation_size') or 47)
    if old_size != 47:
        raise SystemExit(f'{source} observation_size is {old_size}, expected 47')
    encoder = state['encoder.weight']
    if encoder.shape[1] != old_size or tuple(state['obs_mean'].shape) != (old_size,):
        raise SystemExit('Unexpected encoder/obs-stat shapes')
    state['encoder.weight'] = torch.nn.functional.pad(encoder, (0, NEW_DIMS))
    state['obs_mean'] = torch.cat([state['obs_mean'], torch.zeros(NEW_DIMS)])
    state['obs_std'] = torch.cat([state['obs_std'], torch.ones(NEW_DIMS)])
    checkpoint['observation_size'] = old_size + NEW_DIMS
    extra = dict(checkpoint.get('extra') or {})
    extra['expanded_from'] = dict(checkpoint=str(args.checkpoint), sha256=file_sha256(source),
                                  observation_size=old_size, added_dims=NEW_DIMS,
                                  note='new encoder columns zero-initialized; obs_mean 0 / obs_std 1')
    checkpoint['extra'] = extra
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target)
    print(json.dumps(dict(source=str(args.checkpoint), output=str(args.output),
                          source_sha256=file_sha256(source), output_sha256=file_sha256(target),
                          observation_size=old_size + NEW_DIMS)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', default='runs/yumi/best_tall.pt')
    parser.add_argument('--output', default='runs/yumi_obs50/best_tall_obs50.pt')
    main(parser.parse_args())
