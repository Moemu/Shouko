"""CPU/stdlib-only regression: python -m app.test_training_control."""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from app.training_control import TrainingControl


class ControlTest(unittest.TestCase):
    def test_unmanaged_stdin_is_ignored(self):
        with patch.dict(os.environ, {}, clear=True):
            control = TrainingControl(10, io.StringIO('finish\n'))
        self.assertIsNone(control._reader)
        self.assertTrue(control.safe_point(lambda phase: None))
        self.assertIsNone(control.fields()['run_id'])

    def test_subprocess_pause_resume_finish(self):
        code = '''
import json, time
from app.training_control import TrainingControl
c = TrainingControl(30)
def status(phase):
    print(json.dumps(dict(phase=phase or 'training', **c.fields())), flush=True)
status('ready')
while c.safe_point(status):
    time.sleep(0.01)
status('stopped')
'''
        process = subprocess.Popen([sys.executable, '-u', '-c', code],
                                   cwd=Path(__file__).resolve().parents[1],
                                   env=dict(os.environ, FLYBODY_RUN_ID='helper-test'),
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        rows = queue.Queue()
        def read():
            for line in process.stdout:
                rows.put(json.loads(line))
        threading.Thread(target=read, daemon=True).start()
        def send(command):
            process.stdin.write(command + '\n')
            process.stdin.flush()
        def phase(wanted):
            for _ in range(10):
                row = rows.get(timeout=5)
                if row['phase'] == wanted:
                    return row
            self.fail(f'Missing phase {wanted}')
        try:
            self.assertEqual(phase('ready')['run_id'], 'helper-test')
            send('pause')
            first, second = phase('paused'), phase('paused')
            self.assertGreater(second['paused_seconds'], first['paused_seconds'] + 0.5)
            self.assertAlmostEqual(first['active_elapsed'], second['active_elapsed'], places=5)
            self.assertGreater(second['heartbeat_at'], first['heartbeat_at'])
            self.assertAlmostEqual(second['wall_elapsed'], second['active_elapsed'] + second['paused_seconds'])
            send('resume')
            resumed = phase('training')
            self.assertGreaterEqual(resumed['paused_seconds'], second['paused_seconds'])
            send('pause')
            phase('paused')
            send('finish')
            phase('stopping')
            phase('stopped')
            self.assertEqual(process.wait(timeout=5), 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stdin.close()
            process.stdout.close()
            process.stderr.close()

    def test_checkpoint_pair_without_gpu_imports(self):
        # Execute the real checkpoint helpers with a JSON serializer and fake
        # optimizer, not torch or a model/weights from the project.
        source = Path(__file__).with_name('train_full.py').read_text()
        tree = ast.parse(source)
        names = {'file_sha256', 'atomic_model_save', 'validated_checkpoint_state', 'restore_optimizer'}
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        class Serializer:
            fail = False
            @classmethod
            def save(cls, value, path):
                if cls.fail:
                    raise OSError('simulated failed state write')
                Path(path).write_text(json.dumps(value))
            @staticmethod
            def load(path, **kwargs):
                return json.loads(Path(path).read_text())
            @staticmethod
            def is_tensor(value):
                return False
        class Model:
            def save(self, path, extra, physics=None):
                Path(path).write_text(json.dumps(extra))
        class Optimizer:
            param_groups = [{'params': [object()]}]
            loaded = None
            def load_state_dict(self, value):
                self.loaded = value
        scope = dict(Path=Path, hashlib=hashlib, torch=Serializer)
        exec(compile(tree, 'train_full.py', 'exec'), scope)
        with tempfile.TemporaryDirectory() as folder:
            model, state = Path(folder)/'best.pt', Path(folder)/'ppo_state_best.pt'
            save, load = scope['atomic_model_save'], scope['validated_checkpoint_state']
            save(Model(), model, {'updates': 0}, {'value': 'initial'}, state)
            self.assertEqual(load(model, state)['value'], 'initial')
            old_model, old_state = model.read_bytes(), state.read_bytes()
            Serializer.fail = True
            with self.assertRaises(OSError):
                save(Model(), model, {'updates': 1}, {}, state)
            self.assertEqual(model.read_bytes(), old_model)
            self.assertEqual(state.read_bytes(), old_state)
            Serializer.fail = False
            model.write_text('changed')
            with self.assertRaises(ValueError):
                load(model, state)
        optimizer = Optimizer()
        saved = dict(param_groups=[dict(params=[0])], state={})
        self.assertTrue(scope['restore_optimizer'](optimizer, saved, [['frozen']], [['frozen']]))
        self.assertIs(optimizer.loaded, saved)
        self.assertFalse(scope['restore_optimizer'](optimizer, saved, [['frozen']], [['unfrozen']]))
        self.assertFalse(scope['restore_optimizer'](optimizer, dict(param_groups=[], state={})))


if __name__ == '__main__':
    unittest.main()
