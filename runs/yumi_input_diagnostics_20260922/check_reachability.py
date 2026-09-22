"""Count sensory rows that can influence a motor row within the unroll depth."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


path = Path('data/full_graph.npz')
with path.open('rb') as handle:
    digest = hashlib.file_digest(handle, 'sha256').hexdigest()
metadata = json.loads(Path('data/full_graph.json').read_text())
assert digest == metadata['graph_sha256']
with np.load(path, allow_pickle=False) as graph:
    inputs, outputs = graph['inputs'], graph['outputs']
    size = len(graph['ids'])
    # Rows are postsynaptic, columns presynaptic. Transpose walks backward from outputs.
    adjacency = csr_matrix((graph['weight'] != 0, graph['pre'], graph['ptr']), shape=(size, size))
reachable = np.zeros(size, dtype=bool)
reachable[outputs] = True
history = []
for hops in range(9):
    active = reachable[inputs]
    history.append(dict(max_edges=hops, neural_steps_needed=hops+1,
                        input_rows_reachable=int(active.sum()), input_rows=len(inputs),
                        fraction=float(active.mean())))
    reachable |= adjacency.T @ reachable
report = dict(kind='structural_input_motor_reachability', graph_sha256=digest,
              current_neural_steps=4, history=history,
              explanation='State resets to zero. First neural update injects sensory drive; four updates permit at most three edges. Residual state and repeated injection permit shorter paths.',
              limitation='Structural upper bound for this zero-state unroll, not physiological reachability or proof of control quality. Nonlinear gain and readout cancellation may further reduce influence.')
output = Path('runs/yumi_input_diagnostics_20260922/reachability.json')
output.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
