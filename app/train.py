"""Train the fly motor readout, then select using held-out physical walking episodes."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from .brain import FlyBrain, sensation, ROOT
from .sim import Body

RUNS = ROOT / "runs"


def evaluate(brain, body, seed, speed=0.5, seconds=20, record=None):
    body.reset(seed)
    start = time.perf_counter()
    errors=[]; heights=[]; upright=[]; frames=[]
    cmd=np.zeros(3)
    for step in range(round(seconds/0.02)):
        if step%5 == 0:
            cmd = brain.command(sensation(body,speed))
        state = body.step(cmd)
        if state["time"]>2:
            errors.append((state["velocity"][0]-speed)**2)
        heights.append(state["height"]); upright.append(state["upright"])
        if record and step%2 == 0:
            frames.append({**state,"command":cmd.tolist()})
        if state["fallen"]:
            break
    elapsed=state["time"]
    mean_speed=float(body.data.qpos[0]/elapsed)
    rmse=float(np.sqrt(np.mean(errors))) if errors else 10.0
    success=bool(not state["fallen"] and elapsed>=seconds-0.01 and abs(mean_speed-speed)<0.2 and min(state["foot_strikes"])>=4)
    result={"seed":seed,"target_speed":speed,"seconds":elapsed,"mean_speed":mean_speed,
        "forward_m":float(body.data.qpos[0]),"lateral_m":float(body.data.qpos[1]),
        "speed_rmse":rmse,"minimum_height":min(heights),"minimum_upright":min(upright),
        "foot_strikes":state["foot_strikes"],"fallen":state["fallen"],"success":success,
        "score":float(elapsed/seconds*2-rmse-abs(body.data.qpos[1])/max(elapsed,1)),
        "wall_seconds":time.perf_counter()-start}
    if record:
        Path(record).write_text(json.dumps({"meta":result,"frames":frames}))
    return result


def write_status(status):
    target=RUNS/"training.json"
    tmp=target.with_suffix(".tmp")
    tmp.write_text(json.dumps(status,indent=2))
    tmp.replace(target)


def training_labels(x):
    # Engineered teacher for descending velocity commands; never imported by live inference.
    return np.column_stack([np.clip(x[:,0]+0.35*(x[:,0]-x[:,1]),0,0.85),
        np.zeros(len(x)),np.clip(1.4*x[:,2]-0.18*x[:,3],-0.65,0.65)])


def train(samples=1536):
    threadpool_limits(1)
    brain=FlyBrain(); body=Body()
    start=time.perf_counter(); history=[]
    rng=np.random.default_rng(101)
    total=samples+256
    X=np.column_stack([rng.uniform(0,0.75,total),rng.uniform(-0.1,0.9,total),
        rng.uniform(-0.45,0.45,total),rng.uniform(-0.5,0.5,total),
        rng.uniform(-0.15,0.15,total),rng.uniform(-0.12,0.12,total),
        rng.uniform(-0.08,0.05,total),np.ones(total)]).astype(np.float32)
    Y=training_labels(X)
    # Validation draws use a separate seed and are never used to fit weights.
    vrng=np.random.default_rng(202)
    V=vrng.uniform(np.min(X,axis=0),np.max(X,axis=0),(256,8)).astype(np.float32)
    V[:,-1]=1; VY=training_labels(V)
    features=[]
    for i,x in enumerate(np.concatenate([X[:samples],V])):
        features.append(brain.encode(x))
        if i%64==0:
            status={"phase":"encoding","sample":i,"total":samples+256,"elapsed":time.perf_counter()-start,"history":history}
            write_status(status); print(json.dumps(status),flush=True)
    F=np.asarray(features[:samples]); VF=np.asarray(features[samples:])
    brain.scale=np.maximum(np.std(F,axis=0),1e-6).astype(np.float32)
    F=F/brain.scale; VF=VF/brain.scale
    np.savez_compressed(RUNS/"training_features.npz",features=F,targets=Y[:samples],validation=VF,validation_targets=VY)
    initial=evaluate(brain,body,202,seconds=12)
    history.append({"samples":0,"validation_mse":float(np.mean(VY**2)),"evaluation":initial})
    best_score=-float("inf")
    for count in sorted(set([min(n,samples) for n in [64,128,256,512,1024,samples]])):
        f=F[:count].astype(np.float64)
        ridge=0.03
        brain.readout=solve(f.T@f+ridge*np.eye(f.shape[1]),f.T@Y[:count],assume_a="pos").astype(np.float32)
        mse=float(np.mean((VF@brain.readout-VY)**2))
        physical=evaluate(brain,body,202,seconds=20)
        entry={"samples":count,"validation_mse":mse,"evaluation":physical}
        history.append(entry)
        brain.save(RUNS/f"readout_{count}.npz")
        if physical["score"]>best_score:
            best_score=physical["score"]
            brain.save(RUNS/"best.npz")
        status={"phase":"fitting","sample":count,"total":samples,"history":history,"elapsed":time.perf_counter()-start}
        write_status(status); print(json.dumps(entry),flush=True)
    brain.load(RUNS/"best.npz")
    cases=[(s,v) for s,v in zip(range(1001,1010),[0.35,0.5,0.65]*3)]
    results=[evaluate(brain,body,s,v,seconds=30) for s,v in cases]
    controls={}
    for lesion in ["disconnected","rewired"]:
        control=FlyBrain(RUNS/"best.npz",lesion=lesion)
        controls[lesion]=[evaluate(control,body,s,0.5,seconds=30) for s in [1101,1102,1103]]
    final={"graph":brain.meta,"checkpoint_sha256":brain.checkpoint_hash,
        "training_seed":101,"validation_seed":202,"test_seeds":[s for s,v in cases],
        "method":"supervised ridge readout; fixed measured rate reservoir; frozen pretrained Unitree motor policy",
        "criterion":"30 s without fall, mean forward speed within 0.2 m/s of target, >=4 ground contacts per foot",
        "tests":results,"controls":controls,"successes":sum(r["success"] for r in results),"attempts":len(results),
        "elapsed":time.perf_counter()-start,"history":history,
        "limitations":["8192-neuron induced subgraph, not full brain","engineered sensation and signed rate dynamics",
            "pretrained 12-joint G1 motor policy, not learned biological VNC","VRM retargeting, G1 collision and inertia model",
            "no evidence of fly-specific advantage over a matched trained random network"]}
    (RUNS/"evaluation.json").write_text(json.dumps(final,indent=2))
    evaluate(brain,body,1201,seconds=30,record=RUNS/"replay.json")
    write_status({"phase":"complete","sample":samples,"total":samples,"history":history,"elapsed":time.perf_counter()-start})
    print(json.dumps({"successes":final["successes"],"attempts":len(results),"elapsed":final["elapsed"]}),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--samples",type=int,default=1536)
    args=parser.parse_args()
    if args.samples<64 or args.samples>16384:
        parser.error("samples must be 64..16384")
    train(args.samples)
