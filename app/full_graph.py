"""Build the complete retained MaleCNS graph, with no degree or weight pruning."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]


def build():
    source = ROOT / 'data/connectome_data/malecns_v1/normalized'
    report = json.loads((source / 'report.json').read_text())
    nodes = feather.read_table(source / 'neurons.feather')
    reader = ipc.open_file(pa.memory_map(str(source / 'edges.arrow'), 'r'))
    columns = [[], [], []]
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        for j in range(3):
            columns[j].append(batch.column(j).to_numpy())
    pre, post, count = [np.concatenate(parts) for parts in columns]
    n = len(nodes)
    matrix = csr_matrix((count, (post, pre)), shape=(n, n))
    matrix.sum_duplicates()
    matrix.sort_indices()
    assert matrix.nnz == report['graph']['retained_edge_rows']
    assert int(matrix.sum()) == report['graph']['retained_synaptic_contacts']
    transmitter = np.array([str(value or 'missing').lower() for value in nodes['neurotransmitter'].to_pylist()])
    signs = np.where(np.isin(transmitter, ['gaba', 'glutamate', 'histamine']), -1, 1).astype(np.int8)
    ambiguous = ~np.isin(transmitter, ['acetylcholine', 'gaba', 'glutamate', 'histamine'])
    superclass = nodes['superclass'].to_numpy()
    inputs = np.flatnonzero(np.isin(superclass, ['vnc_sensory', 'cb_sensory', 'ol_sensory']))
    outputs = np.flatnonzero(np.isin(superclass, ['vnc_motor', 'cb_motor']))
    values = np.sqrt(matrix.data.astype(np.float32))
    rows = np.repeat(np.arange(n), np.diff(matrix.indptr))
    incoming = np.bincount(rows, weights=values, minlength=n)
    values *= signs[matrix.indices] / np.maximum(incoming[rows], 1)
    target = ROOT / 'data/full_graph.npz'
    np.savez_compressed(target, ids=nodes['source_id'].to_numpy().astype(np.uint64),
                        ptr=matrix.indptr.astype(np.int32), pre=matrix.indices.astype(np.int32),
                        weight=values, inputs=inputs, outputs=outputs,
                        coords=np.column_stack([nodes[name].to_numpy() for name in ['soma_x', 'soma_y', 'soma_z']]).astype(np.float32))
    with target.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    metadata = dict(dataset='MaleCNS v1.0', neurons=n, edges=matrix.nnz,
                    contacts=int(matrix.sum()), input_neurons=len(inputs), output_neurons=len(outputs),
                    ambiguous_sign_neurons=int(ambiguous.sum()), graph_sha256=digest,
                    source_hashes=report['source_hashes'],
                    selection='Every retained neuron and every released connection between them; no extra pruning.',
                    input_mapping='Learned body-observation encoder into annotated sensory neurons.',
                    output_mapping='Learned joint readout from annotated VNC and CB motor neurons.',
                    dynamics='Engineered signed rate network; not a validated biological spiking model.')
    (ROOT / 'data/full_graph.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    build()
