"""CPU fixture checks for runtime intervention semantics; no checkpoint updates."""
import torch
from app.full_brain import ConnectomePolicy
from app.probe_feedback_control import FeedbackIntervention
from app.test_policy_contract import tiny_graph

torch.set_num_threads(2)
torch.manual_seed(21)
with tiny_graph(), torch.inference_mode():
    policy=ConnectomePolicy(observation_size=50,device='cpu')
    model=FeedbackIntervention(policy)
    observations=torch.randn(8,50)
    original=observations.clone()
    weights=policy.encoder.weight.clone()
    torch.testing.assert_close(model(observations)[0],policy(observations)[0],rtol=0,atol=0)
    model.zero_new_inputs=True
    changed=original.clone()
    changed[:,47:]=0
    torch.testing.assert_close(model(observations)[0],policy(changed)[0],rtol=0,atol=0)
    torch.testing.assert_close(observations,original,rtol=0,atol=0)
    torch.testing.assert_close(policy.encoder.weight,weights,rtol=0,atol=0)
    policy.encoder.weight[:,47:]=0
    torch.testing.assert_close(model(observations)[0],policy(observations)[0],rtol=0,atol=0)
print('PASS: intervention changes only new observations, preserves inputs/weights, zero-column control is equivalent')
