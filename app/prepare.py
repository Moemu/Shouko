"""Stream MaleCNS data; keep a measured, reproducible sensorimotor subgraph."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
os.environ["FLYCUBE_DATA_DIR"] = str(ROOT / "data")
sys.path.insert(0, str(ROOT / "vendor/flycube"))
from flycube.connectome.download import ensure
from flycube.connectome.import_malecns import import_malecns, OUTPUT_DIR
from flycube.connectome.transmitters import sign_from_nt


def digest(path):
    with open(path,"rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def build():
    ensure()
    if not (OUTPUT_DIR / "report.json").exists():
        print(json.dumps(import_malecns()),flush=True)
    nodes = feather.read_table(OUTPUT_DIR / "neurons.feather").to_pandas()
    report = json.loads((OUTPUT_DIR / "report.json").read_text())
    print(report["superclass_counts"],flush=True)
    n = len(nodes)
    reader = ipc.open_file(pa.memory_map(str(OUTPUT_DIR / "edges.arrow"), "r"))
    degree = np.zeros(n,np.int64)
    for k in range(reader.num_record_batches):
        b = reader.get_batch(k)
        for col in [0,1]:
            degree += np.bincount(b.column(col).to_numpy(),minlength=n)
    sc = nodes.superclass.astype(str).to_numpy()
    inputs = np.flatnonzero(sc == "visual_projection")
    outputs = np.flatnonzero(sc == "descending_neuron")
    if not len(inputs) or not len(outputs):
        raise ValueError("Missing biological input or descending groups")
    def ranked(indices, count):
        return indices[np.lexsort((indices,-degree[indices]))[:count]]
    chosen_input = ranked(inputs,768)
    chosen_output = ranked(outputs,384)
    chosen = np.union1d(chosen_input,chosen_output)
    # ponytail: degree-ranked anatomical subgraph; no full CSR allocation on 16 GB hosts.
    other = np.setdiff1d(np.flatnonzero(sc == "cb_intrinsic"),chosen)
    chosen = np.union1d(chosen,ranked(other,8192-len(chosen)))
    if len(chosen) < 8192:
        chosen = np.union1d(chosen,ranked(np.setdiff1d(np.arange(n),chosen),8192-len(chosen)))
    index = np.full(n,-1,np.int32)
    index[chosen] = np.arange(len(chosen))
    pre_parts, post_parts, count_parts = [], [], []
    for k in range(reader.num_record_batches):
        b = reader.get_batch(k)
        p = index[b.column(0).to_numpy()]
        q = index[b.column(1).to_numpy()]
        keep = (p>=0)&(q>=0)
        if np.any(keep):
            pre_parts.append(p[keep]); post_parts.append(q[keep])
            count_parts.append(b.column(2).to_numpy()[keep])
    matrix = csr_matrix((np.concatenate(count_parts),(np.concatenate(pre_parts),np.concatenate(post_parts))),shape=(len(chosen),len(chosen)))
    matrix.sum_duplicates(); matrix.sort_indices()
    selected = nodes.iloc[chosen]
    input_idx = np.flatnonzero(selected.superclass.to_numpy() == "visual_projection")
    output_idx = np.flatnonzero(selected.superclass.to_numpy() == "descending_neuron")
    reach = np.zeros(len(chosen),bool); reach[input_idx] = True
    adjacency = matrix.copy().astype(np.float32); adjacency.data[:] = 1
    for _ in range(12):
        reach |= (adjacency.T @ reach.astype(np.float32)) > 0
    output_idx = output_idx[reach[output_idx]]
    assert len(output_idx)>32 and not np.intersect1d(input_idx,output_idx).size
    signs, ambiguous = sign_from_nt(selected.neurotransmitter)
    target = ROOT / "data/graph.npz"
    np.savez_compressed(target,ids=selected.source_id.to_numpy(np.uint64),
        ptr=matrix.indptr,post=matrix.indices,contact=matrix.data,
        signs=signs,input_idx=input_idx,output_idx=output_idx,
        coords=selected[["soma_x","soma_y","soma_z"]].to_numpy(np.float32))
    meta = {"dataset":"MaleCNS v1.0", "neurons":len(chosen), "edges":matrix.nnz,
        "contacts":int(matrix.data.sum()),"full_neurons":n,
        "full_edges":report["graph"]["retained_edge_rows"],
        "input_neurons":len(input_idx),"output_neurons":len(output_idx),
        "ambiguous_sign_neurons":int(ambiguous.sum()),
        "selection":"Top total-degree: 768 visual_projection + 384 descending_neuron; fill to 8192 with cb_intrinsic neurons; integer ID breaks ties. Keep all induced measured edges. Readout uses descending neurons reachable within 12 hops.",
        "graph_sha256":digest(target),"source_hashes":report["source_hashes"],
        "coordinate_space":"measured soma voxel anchors; not neuron morphology",
        "dynamics":"chosen signed tanh rate deviations, not LIF or biological spikes"}
    (ROOT / "data/graph.json").write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)


if __name__ == "__main__":
    build()
