"""Small CPU regressions for PPO returns, checkpoint identity and the warmup gate."""
import hashlib
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from app.ppo_yumi import same_weights, unchanged_checkpoint, value_gate_ready
from app.value_diagnostics import diagnose, gae_targets, regression_metrics, save_rollout


def check_returns():
    rewards = torch.full((128, 2), 2.0, dtype=torch.float64)
    values = torch.full((129, 2), 200.0, dtype=torch.float64)
    dones = torch.zeros_like(rewards)
    advantages, returns = gae_targets(rewards, values, dones, 0.99, 0.95)
    torch.testing.assert_close(advantages, torch.zeros_like(advantages))
    torch.testing.assert_close(returns, values[:-1])
    # A terminal reward cannot see a large value from the newly reset episode.
    _, returns = gae_targets(torch.tensor([[1.], [2.]]),
                            torch.tensor([[3.], [999.], [1000.]]),
                            torch.tensor([[1.], [0.]]), 0.9, 1.0)
    torch.testing.assert_close(returns, torch.tensor([[1.], [902.]]))
    metrics = regression_metrics(torch.tensor([11., 12., 13.]), torch.tensor([1., 2., 3.]))
    assert metrics['explained_variance'] == 1 and metrics['r2'] < 0
    assert regression_metrics(torch.ones(3), torch.ones(3))['r2'] is None


def check_gate():
    assert not value_gate_ready([1.] * 5, 50)
    assert value_gate_ready([1.] * 6, 50)
    assert not value_gate_ready([300., 1, 1, 1, 1, 1], 50)  # Only final window passes.
    assert not value_gate_ready([50.] * 6, 50)
    assert not value_gate_ready([1.] * 5 + [float('nan')], 50)


def check_identity_and_export():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root/'best.pt'
        state = dict(weight=torch.arange(5), bias=torch.zeros(2))
        torch.save(dict(state_dict=state, extra=dict(combined_bar=0.869)), checkpoint)
        before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert unchanged_checkpoint(checkpoint, state)
        assert not same_weights(state, dict(state, bias=torch.ones(2)))
        assert not same_weights(state, dict(weight=state['weight']))
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
        path = root/'rollout.pt'
        rewards, values = torch.ones(3, 2), torch.zeros(4, 2)
        dones = torch.tensor([[0., 1.], [0., 0.], [1., 0.]])
        advantages, returns = gae_targets(rewards, values, dones, 0.99, 0.95)
        save_rollout(path, observations=torch.zeros(3, 2, 50), actions=torch.zeros(3, 2, 12),
                     rewards=rewards, dones=dones, values=values, advantages=advantages, returns=returns,
                     std=torch.full((12,), 0.08), gamma=0.99, lam=0.95,
                     interface=dict(cmd_scale=[2, 2, 0.25]), iteration=0, seed=7,
                     checkpoint_sha256='synthetic-fixture', reward_config={})
        batch = torch.load(path, map_location='cpu', weights_only=True)
        result = diagnose(batch)
        assert result['all']['samples'] == 6 and result['terminal']['samples'] == 2
        assert abs(result['exploration_std'][0] - 0.08) < 1e-7
        assert result['per_command'][0]['samples'] == 6
        batch['values'][0, 0] = float('nan')
        try:
            diagnose(batch)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid rollout accepted')


def check_training_selection():
    """Execute the real selection branches without constructing a GPU trainer."""
    source = Path(__file__).with_name('ppo_yumi.py')
    tree = ast.parse(source.read_text(encoding='utf-8'))
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    startup = next(n for n in train.body if isinstance(n, ast.If) and
                   isinstance(n.test, ast.Name) and n.test.id == 'same_as_best')
    loop = next(n for n in train.body if isinstance(n, ast.While))
    evaluation = next(n for n in loop.body if isinstance(n, ast.If) and
                      ast.unparse(n.test) == 'iteration % args.eval_every == 0')
    first = next(i for i, n in enumerate(evaluation.body) if isinstance(n, ast.Expr) and
                 isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and
                 n.value.func.id == 'save_checkpoint')
    last = next(i for i, n in enumerate(evaluation.body) if isinstance(n, ast.Assign) and
                ast.unparse(n.targets[0]) == "status['evaluation']")
    with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
        root = Path(directory)
        path = root/'best.pt'
        state = dict(weight=torch.zeros(3))
        torch.save(dict(state_dict=state), path)
        before = path.read_bytes()
        saves = []
        context = dict(same_as_best=True, stored_bar=0.869, base=dict(score=1.5),
                       combined=lambda ev: ev['score'], json=json, RUNS=root, iteration=5,
                       policy=SimpleNamespace(state_dict=lambda: state), status={},
                       save_checkpoint=lambda name, *args, **kwargs: saves.append(name),
                       unchanged_checkpoint=unchanged_checkpoint)
        exec(compile(ast.Module(body=[startup], type_ignores=[]), str(source), 'exec'), context)
        assert not saves and context['best_combined'] == 0.869
        context['result'] = dict(score=1.5)
        exec(compile(ast.Module(body=evaluation.body[first:last+1], type_ignores=[]),
                     str(source), 'exec'), context)
        assert saves == ['last'] and context['best_combined'] == 0.869
        assert path.read_bytes() == before
        # A genuinely changed candidate still uses confirmation and can win.
        state['weight'] = torch.ones(3)
        context.update(evaluate=lambda *args, **kwargs: dict(score=1.4), env=None,
                       args=SimpleNamespace(eval_seconds=1, ratchet_margin=0.03))
        exec(compile(ast.Module(body=evaluation.body[first:last+1], type_ignores=[]),
                     str(source), 'exec'), context)
        assert saves == ['last', 'last', 'best']
        assert abs(context['best_combined'] - 1.45) < 1e-8


if __name__ == '__main__':
    check_returns()
    check_gate()
    check_identity_and_export()
    check_training_selection()
    print('PPO diagnostics: GAE, terminal masks, value metrics, warmup gate, identity, export OK')
