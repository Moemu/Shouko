"""Build an explicitly derived actor/critic initialization, not a training resume."""
import hashlib
import json
from pathlib import Path
import shutil
import torch
from app.train_full import validated_checkpoint_state

root = Path('runs/direction_control_20260929')
actor = root/'transfer_main/round4/last.pt'
teacher = root/'teacher_control3029/last.pt'
folder = root/'readout_warm_initial'
folder.mkdir(exist_ok=False)
state = validated_checkpoint_state(teacher, teacher.with_name('ppo_state.pt'), device='cpu')
assert 'value_observation_stats' in state
shutil.copyfile(actor, folder/'last.pt')
def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
provenance = dict(kind='derived_initialization_not_joint_training',
    actor_checkpoint_sha256=sha(actor), critic_checkpoint_sha256=sha(teacher),
    critic_state_sha256=sha(teacher.with_name('ppo_state.pt')),
    actor_optimizer='fresh', exploration_std=0.05,
    critic_optimizer='teacher', critic_observation_stats='teacher')
assert sha(folder/'last.pt') == provenance['actor_checkpoint_sha256']
derived = dict(value=state['value'], value_optimizer=state['value_optimizer'],
    value_observation_stats=state['value_observation_stats'],
    optimizer=None, optimizer_names=None,
    log_std=torch.full((12,), float(torch.log(torch.tensor(.05)))),
    checkpoint_sha256=provenance['actor_checkpoint_sha256'], initialization_provenance=provenance)
torch.save(derived, folder/'ppo_state.pt')
validated_checkpoint_state(folder/'last.pt', folder/'ppo_state.pt', device='cpu')
(folder/'provenance.json').write_text(json.dumps(provenance, indent=2))
print(json.dumps(provenance))
