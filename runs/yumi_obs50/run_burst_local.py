"""Resource-gated local PPO burst runner for the 50-dim upright lineage.

This laptop shares one 8 GB GPU with the desktop (WDDM) and had a driver-level
GPU crash on 2026-09-20 while memory-saturated, so every phase gates on free
system RAM and free VRAM and waits until both are safe:

  0. wait for gates (RAM >= 4 GiB available, VRAM >= 5 GiB free)
  1. warm-start equivalence check, one short process per config
  2. footprint probe: one 90 s ppo_yumi run at the training worlds count;
     abort unless peak VRAM + 2 GiB desktop margin fits in 8 GiB
  3. burst loop: ppo15 ratchet recipe (PPO_POSTURE_20260916), 16 worlds,
     each burst resuming <runs-dir>/best.pt (+ optimizer state when present)

Run with .venv-gpu:  python runs/yumi_obs50/run_burst_local.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON = sys.executable
LOG = HERE / 'launcher.log'
RAM_GATE_GIB = 4.0
VRAM_GATE_MIB = 5 * 1024
WORLDS = 16
BURSTS = 8
BURST_SECONDS = 420
GATE_TIMEOUT_S = 2 * 3600


def log(message):
    line = json.dumps(dict(t=time.strftime('%H:%M:%S'), msg=message), ensure_ascii=False)
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as handle:
        handle.write(line + '\n')


def free_vram_mib():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, timeout=30)
    return int(out.stdout.strip().splitlines()[0])


def gates_open():
    ram_gib = psutil.virtual_memory().available / 2**30
    vram_mib = free_vram_mib()
    ok = ram_gib >= RAM_GATE_GIB and vram_mib >= VRAM_GATE_MIB
    return ok, f'ram_free={ram_gib:.1f}GiB vram_free={vram_mib}MiB'


def wait_for_gates(purpose):
    started = time.time()
    while True:
        ok, detail = gates_open()
        if ok:
            log(f'gates open for {purpose}: {detail}')
            return True
        if time.time() - started > GATE_TIMEOUT_S:
            log(f'gate timeout for {purpose}: {detail}')
            return False
        log(f'waiting for gates ({purpose}): {detail}')
        time.sleep(60)


def run(cmd, purpose):
    log(f'start {purpose}: {" ".join(cmd)}')
    started = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    log(f'finished {purpose}: exit={result.returncode} wall={time.time()-started:.0f}s')
    return result.returncode == 0


def base_args(resume, runs_dir):
    return [PYTHON, '-m', 'app.ppo_yumi',
            '--resume', resume, '--runs-dir', runs_dir,
            '--interface-yaml', 'runs/local/yumi_tall.yaml',
            '--freeze-brain', '--lr', '1e-4', '--std-override', '0.08',
            '--epochs', '4', '--target-kl', '0.02', '--value-warmup', '6',
            '--eval-every', '5', '--worlds', str(WORLDS), '--knee-gate', 'stance']


def main():
    log(f'launcher start: worlds={WORLDS} bursts={BURSTS} burst_seconds={BURST_SECONDS}')

    if not wait_for_gates('warmstart check'):
        return 1
    for config in ['source_47', 'expanded_50']:
        if not run([PYTHON, 'runs/yumi_obs50/verify_warmstart.py', '--config', config,
                    '--worlds', str(WORLDS), '--seconds', '10'], f'warmstart {config}'):
            log('warm-start check failed; aborting')
            return 1
    verdict = json.load(open(HERE / 'warmstart_check.json')).get('verdict', {})
    log(f'warmstart verdict: {json.dumps(verdict)}')
    if not verdict.get('bit_equivalent'):
        log('ABORT: expanded checkpoint is not behaviour-equivalent to the source')
        return 1

    if not wait_for_gates('footprint probe'):
        return 1
    if not run(base_args(str(HERE / 'best_tall_obs50.pt'), str(HERE / 'probe')) +
               ['--max-seconds', '90', '--value-warmup', '1', '--eval-seconds', '10'],
               'footprint probe'):
        log('probe crashed; aborting (GPU state suspect after the earlier driver crash)')
        return 1
    peak_gib = json.load(open(HERE / 'probe' / 'training.json')).get('peak_vram_gib', 0)
    total_mib = free_vram_mib()  # after probe exit, this is close to the desktop-only usage
    projected = peak_gib * 1024 + (8188 - total_mib)
    log(f'probe peak_vram={peak_gib:.2f}GiB; projected total={projected:.0f}MiB of 8188MiB')
    if projected > 7300:  # keep ~0.85 GiB headroom for the desktop compositor
        log('ABORT: projected VRAM usage leaves too little desktop headroom; '
            'close desktop apps or lower --worlds')
        return 1

    for burst in range(1, BURSTS + 1):
        if not wait_for_gates(f'burst {burst}'):
            return 1
        if burst == 1 or not (HERE / 'ppo_state_best.pt').exists():
            cmd = base_args(str(HERE / 'best_tall_obs50.pt'), str(HERE))
        else:
            cmd = base_args(str(HERE / 'best.pt'), str(HERE)) + ['--resume-state']
        cmd += ['--max-seconds', str(BURST_SECONDS)]
        if not run(cmd, f'burst {burst}'):
            log(f'burst {burst} failed; stopping the loop (best.pt keeps the ratcheted checkpoint)')
            return 1
    log('all bursts complete')
    return 0


if __name__ == '__main__':
    sys.exit(main())
