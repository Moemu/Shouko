"""One numerical and evidence check for the training/inference boundary."""
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .brain import FlyBrain, ROOT


def main():
    # Edges are pre -> post. Unit input at node 0 must reach node 1, not the reverse.
    matrix=csr_matrix(([2.0],([1],[0])),shape=(2,2))
    assert np.array_equal(matrix@np.array([1.0,0]),[0,2])
    brain=FlyBrain(ROOT/"runs/best.npz")
    x=np.array([0.5,0.45,0.02,0,0,0,0,1],np.float32)
    first=brain.command(x); second=brain.command(x)
    assert np.array_equal(first,second) and np.all(np.isfinite(first))
    assert first[0]>0.3
    assert not np.intersect1d(brain.inputs,brain.outputs).size
    assert len(np.unique(brain.ids))==8192 and brain.ids.dtype.kind=='u'
    disconnected=FlyBrain(ROOT/"runs/best.npz",lesion="disconnected")
    assert np.array_equal(disconnected.command(x),np.zeros(3))
    assert np.count_nonzero(disconnected.activity[disconnected.outputs])==0
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/"roundtrip.npz"
        brain.save(path); restored=FlyBrain(path)
        assert np.array_equal(restored.command(x),first)
    evaluation=json.loads((ROOT/"runs/evaluation.json").read_text())
    assert evaluation["checkpoint_sha256"]==brain.checkpoint_hash
    assert evaluation["successes"]==9 and evaluation["attempts"]==9
    assert all(not r["fallen"] and r["seconds"]>29.99 for r in evaluation["tests"])
    assert not any(r["success"] for r in evaluation["controls"]["disconnected"])
    assert evaluation["history"][-1]["validation_mse"] < evaluation["history"][0]["validation_mse"]/100
    assert evaluation["validation_seed"] not in evaluation["test_seeds"]
    result={"passed":True,"checks":["pre-to-post propagation","determinism","finite action",
        "exact neuron IDs","disjoint sensory/motor groups","zero disconnected output",
        "checkpoint roundtrip","checkpoint-evidence hash match","held-out walking evidence",
        "learning improvement","separate validation/test seeds"]}
    (ROOT/"runs/checks.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
