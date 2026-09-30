"""Separate CUDA CSR repeatability from the selected-column initialization contract."""
import json
from pathlib import Path
import torch
from app.evaluate_locomotion import load_policy
from app.imitate_yumi import dataset
from app.ordered_inference import ordered_inference
from app.train_sensory_readout import VyEncoder
from app.train_full import file_sha256

torch.set_num_threads(8)
source=Path('runs/direction_control_20260929/transfer_main/round4/last.pt')
fit=json.loads(source.with_name('fit.json').read_text())
data=dataset(fit['arguments']['train']);x=data['observations'][:32].cuda()
model,_,_=load_policy(source)
def native():
    with torch.no_grad():return model(x)[0]
def ordered():
    with ordered_inference():return model(x)[0]
native_a,native_b=native(),native()
ordered_a,ordered_b=ordered(),ordered()
model.obs_std[48]=data['observations'][:,48].std().clamp_min(1e-4)
model.requires_grad_(False);model.readout.requires_grad_(True)
model.encoder=VyEncoder(model.encoder)
native_c,native_d=native(),native()
ordered_c,ordered_d=ordered(),ordered()
report=dict(source_sha256=file_sha256(source),script_sha256=file_sha256(__file__),
    native_repeat_max=float((native_a-native_b).abs().max()),
    ordered_repeat_max=float((ordered_a-ordered_b).abs().max()),
    native_initialization_max=float((native_a-native_c).abs().max()),
    native_after_repeat_max=float((native_c-native_d).abs().max()),
    ordered_initialization_max=float((ordered_a-ordered_c).abs().max()),
    ordered_after_repeat_max=float((ordered_c-ordered_d).abs().max()),
    native_ordered_max=float((native_a-ordered_a).abs().max()))
Path('runs/sensory_access_20260930/initialization_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
