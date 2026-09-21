import json, shutil, subprocess, sys, time
from pathlib import Path
root=Path('runs/yumi_calibration_20260921')
results=[]
for worlds, minibatch in [(32,128),(128,512)]:
    target=root/f'short{worlds}'
    target.mkdir(exist_ok=False)
    for name in ['best.pt','ppo_state_best.pt']:
        shutil.copy2(Path('runs/yumi_obs50')/name,target/name)
    command=[sys.executable,'-u','-m','app.ppo_yumi','--resume','runs/yumi_obs50/best.pt','--resume-state','--runs-dir',str(target),'--worlds',str(worlds),'--steps','128','--minibatch',str(minibatch),'--epochs','4','--freeze-brain','--lr','0.00005','--lr-final-frac','1','--value-warmup','1','--knee-gate','stance','--target-kl','0.02','--eval-seconds','30','--eval-every','7','--max-iterations','7','--max-seconds','360']
    started=time.time()
    with (target/'train.log').open('w') as log:
        done=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=540)
    results.append(dict(worlds=worlds,minibatch=minibatch,command=command[2:],seconds=time.time()-started,returncode=done.returncode))
    (root/'short_trials.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results[-1]),flush=True)
    if done.returncode: raise SystemExit(done.returncode)
