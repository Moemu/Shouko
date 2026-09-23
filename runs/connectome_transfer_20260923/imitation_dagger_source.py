"""Isolated native Yumi demonstrations and full-connectome supervised diagnostics."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from .evaluate_locomotion import load_policy
from .full_brain import ROOT
from .ordered_inference import ordered_inference
from .sim import Body
from .train_full import atomic_json, atomic_model_save, file_sha256


def metadata(args):
    return dict(arguments=vars(args), source_sha256={name: file_sha256(ROOT/'app'/name)
        for name in ['imitate_yumi.py', 'full_brain.py', 'mlp_policy.py', 'sim.py', 'ordered_inference.py']})


def collect(args):
    teacher, cfg, kind = load_policy(args.teacher)
    if kind != 'mlp':
        raise ValueError('Expected the verified MLP teacher')
    student = None
    if args.student:
        student, student_cfg, student_kind = load_policy(args.student)
        if student_kind != 'connectome' or student_cfg != dict(cfg, neural_steps=student_cfg['neural_steps']):
            raise ValueError('Student and teacher interfaces differ')
    bodies = [Body(False, 'yumi', interface=cfg['physics_interface'], observation_size=50) for _ in range(9)]
    dt = bodies[0].model.opt.timestep*bodies[0].cfg['control_decimation']
    observations, actions, episode_ids, rows = [], [], [], []
    started = time.monotonic()
    for cohort in range(args.cohorts):
        seeds = [args.seed_base+cohort*10+i for i in range(9)]
        yaws = [args.yaws[cohort % len(args.yaws)]]*9
        for body, seed, yaw in zip(bodies, seeds, yaws):
            body.reset(seed)
            body.data.qpos[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
            mujoco.mj_forward(body.model, body.data)
        alive = np.ones(9, dtype=bool)
        speeds = [0.35, 0.5, 0.65]*3
        for step in range(round(args.seconds/dt)):
            obs = np.stack([b.motor_observation([v, 0, float(np.clip(-b.observation()[1]*1.4, -.2, .2))])
                            for b, v in zip(bodies, speeds)])
            tensor = torch.from_numpy(obs).cuda()
            with torch.inference_mode():
                target = teacher(tensor)[0].cpu().numpy()
            observations.append(obs[alive].copy())
            actions.append(target[alive].copy())
            episode_ids.append(np.array(seeds, dtype=np.int64)[alive])
            if student is not None:
                with ordered_inference():
                    predicted = student(tensor)[0].cpu().numpy()
                applied = args.beta*target+(1-args.beta)*predicted
            else:
                applied = target
            for i, body in enumerate(bodies):
                if alive[i]:
                    state = body.step_joints(applied[i])
                    if state['fallen'] or step == round(args.seconds/dt)-1:
                        rows.append(dict(seed=seeds[i], initial_yaw=yaws[i], command=speeds[i],
                                         seconds=state['time'], fallen=bool(state['fallen'])))
                    alive[i] = not state['fallen']
            if not alive.any():
                break
    data = dict(observations=torch.from_numpy(np.concatenate(observations)),
                targets=torch.from_numpy(np.concatenate(actions)),
                episode_ids=torch.from_numpy(np.concatenate(episode_ids)),
                physics_interface=cfg['physics_interface'], teacher_sha256=file_sha256(args.teacher),
                student_sha256=file_sha256(args.student) if args.student else None,
                episodes=rows, **metadata(args))
    torch.save(data, args.output)
    atomic_json(Path(args.output).with_suffix('.json'), dict(samples=len(data['observations']),
                episodes=rows, teacher_sha256=data['teacher_sha256'], student_sha256=data['student_sha256'],
                sha256=file_sha256(args.output), wall_seconds=time.monotonic()-started, **metadata(args)))


def dataset(paths):
    chunks = [torch.load(p, map_location='cpu', weights_only=True) for p in paths]
    for chunk in chunks[1:]:
        if chunk['physics_interface'] != chunks[0]['physics_interface'] or chunk['teacher_sha256'] != chunks[0]['teacher_sha256']:
            raise ValueError('Dataset provenance or interface mismatch')
    return dict(observations=torch.cat([c['observations'] for c in chunks]),
                targets=torch.cat([c['targets'] for c in chunks]),
                episode_ids=torch.cat([c['episode_ids'] for c in chunks]),
                physics_interface=chunks[0]['physics_interface'], teacher_sha256=chunks[0]['teacher_sha256'])


def metrics(predicted, target):
    errors = (predicted-target).double()
    mse = errors.square().mean(0)
    variance = target.double().var(0, unbiased=False).clamp_min(1e-10)
    return dict(mse=float(mse.mean()), rmse=float(mse.mean().sqrt()),
                normalized_mse=float((mse/variance).mean()), per_joint_r2=(1-mse/variance).tolist(),
                maximum_absolute_error=float(errors.abs().max()),
                action_clip_fraction=float((predicted.abs()>8).float().mean()))


@torch.no_grad()
def predict(model, observations, batch, features=False):
    values = []
    with ordered_inference():
        for x in observations.split(batch):
            action, activity = model(x.cuda())
            values.append((model.normalizer(activity[model.outputs].T) if features else action).cpu())
    return torch.cat(values)


def fit(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    model, cfg, kind = load_policy(args.source)
    if kind != 'connectome':
        raise ValueError('Student must retain the full connectome')
    train, validation = dataset(args.train), dataset(args.validation)
    if set(train['episode_ids'].tolist()) & set(validation['episode_ids'].tolist()):
        raise ValueError('Validation episodes overlap training episodes')
    if train['physics_interface'] != validation['physics_interface'] or train['teacher_sha256'] != validation['teacher_sha256']:
        raise ValueError('Train and validation provenance differ')
    model.load(args.source, interface=train['physics_interface'])
    report = dict(completed=False, source_checkpoint_sha256=file_sha256(args.source),
                  data_sha256={p:file_sha256(p) for p in args.train+args.validation},
                  teacher_sha256=train['teacher_sha256'], history=[], **metadata(args))
    started = time.monotonic()
    def save_report():
        report['wall_seconds'] = time.monotonic()-started
        atomic_json(out/'fit.json', report)
    report['initial_validation'] = metrics(predict(model, validation['observations'], args.batch), validation['targets'])
    save_report()
    if args.mode == 'readout':
        features = predict(model, train['observations'], args.batch, True).double()
        vx = predict(model, validation['observations'], args.batch, True).double()
        report['initial_training'] = metrics(features@model.readout.weight.detach().cpu().double().T+
                                             model.readout.bias.detach().cpu().double(), train['targets'])
        torch.save(dict(train=features.float(), validation=vx.float()), out/'features.pt')
        mean, scale = features.mean(0), features.std(0).clamp_min(1e-4)
        x = (features-mean)/scale
        x = torch.cat([x, torch.ones(len(x), 1)], 1)
        gram = x.T@x/len(x)
        reg = torch.eye(x.shape[1], dtype=x.dtype)*args.ridge
        reg[-1, -1] = 0
        coefficients = torch.linalg.solve(gram+reg, x.T@train['targets'].double()/len(x))
        with torch.no_grad():
            model.readout.weight.copy_((coefficients[:-1]/scale[:, None]).T.float().cuda())
            model.readout.bias.copy_((coefficients[-1]-mean@ (coefficients[:-1]/scale[:, None])).float().cuda())
        vp = torch.cat([(vx-mean)/scale, torch.ones(len(vx), 1)], 1)@coefficients
        report['history'].append(dict(update=1, training=metrics(x@coefficients, train['targets']),
                                     validation=metrics(vp, validation['targets'])))
        model.save(out/'last.pt', extra=dict(method='fixed full-graph features, ridge readout', ridge=args.ridge),
                   physics=train['physics_interface'])
    else:
        model.edge_delta.requires_grad_(args.mode == 'edges')
        model.normalizer.requires_grad_(False)
        optimizer = torch.optim.Adam([
            dict(params=list(model.encoder.parameters()), lr=args.lr),
            dict(params=[model.neuron_bias], lr=args.lr*.1),
            dict(params=list(model.readout.parameters()), lr=args.lr),
            *([dict(params=[model.edge_delta], lr=args.lr*.1)] if args.mode == 'edges' else [])], foreach=False)
        tx, ty = train['observations'].cuda(), train['targets'].cuda()
        model.train()
        for update in range(1, args.updates+1):
            indices = torch.randint(len(tx), (args.batch,), device='cuda')
            optimizer.zero_grad(set_to_none=True)
            predicted, _ = model(tx[indices])
            loss = (predicted-ty[indices]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite imitation loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5, foreach=False)
            if not torch.isfinite(grad):
                raise FloatingPointError('Non-finite gradient')
            if update == 1:
                report['first_gradient_max'] = {n:float(p.grad.abs().max()) for n,p in model.named_parameters() if p.grad is not None}
            optimizer.step()
            if update == 1 or update % args.report_every == 0 or update == args.updates:
                row = dict(update=update, training_batch_mse=float(loss), gradient_norm=float(grad),
                           validation=metrics(predict(model, validation['observations'], args.batch), validation['targets']))
                report['history'].append(row)
                save_report()
                print(json.dumps(row), flush=True)
        atomic_model_save(model, out/'last.pt', dict(method=args.mode, updates=args.updates),
                          dict(optimizer=optimizer.state_dict()), out/'last_state.pt', physics=train['physics_interface'])
    report['final_validation'] = metrics(predict(model, validation['observations'], args.batch), validation['targets'])
    report.update(completed=True, checkpoint_sha256=file_sha256(out/'last.pt'),
                  peak_memory_bytes=torch.cuda.max_memory_allocated())
    save_report()
    print(json.dumps(dict(completed=True, validation=report['final_validation'], wall_seconds=report['wall_seconds'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    collect_parser = sub.add_parser('collect')
    collect_parser.add_argument('--teacher', required=True)
    collect_parser.add_argument('--student')
    collect_parser.add_argument('--beta', type=float, default=1)
    collect_parser.add_argument('--seconds', type=float, default=12)
    collect_parser.add_argument('--cohorts', type=int, default=3)
    collect_parser.add_argument('--yaws', nargs='+', type=float, default=[0, -.15, .15])
    collect_parser.add_argument('--seed-base', type=int, required=True)
    fit_parser = sub.add_parser('fit')
    fit_parser.add_argument('--source', required=True)
    fit_parser.add_argument('--train', nargs='+', required=True)
    fit_parser.add_argument('--validation', nargs='+', required=True)
    fit_parser.add_argument('--mode', choices=['readout', 'adapters', 'edges'], default='readout')
    fit_parser.add_argument('--ridge', type=float, default=.001)
    fit_parser.add_argument('--lr', type=float, default=.0001)
    fit_parser.add_argument('--updates', type=int, default=500)
    fit_parser.add_argument('--report-every', type=int, default=100)
    fit_parser.add_argument('--batch', type=int, default=128)
    fit_parser.add_argument('--seed', type=int, default=2028)
    for p in [collect_parser, fit_parser]:
        p.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    (collect if args.command == 'collect' else fit)(args)


if __name__ == '__main__':
    main()
