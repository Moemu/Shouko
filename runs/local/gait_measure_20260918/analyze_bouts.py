import numpy as np

base = 'runs/local/gait_measure_20260918/'


def bouts(mask):
    # list of (start, length) for True-runs
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        if not v and s is not None:
            out.append((s, i - s))
            s = None
    if s is not None:
        out.append((s, len(mask) - s))
    return out


for lin, idx, label in [('tall', 3, 'tall  cmd0.5 s8001'), ('squat', 3, 'squat cmd0.5 s8001'),
                        ('tall', 0, 'tall  cmd0.35 s8001')]:
    f = np.load(base + f'gait_{lin}_frames.npz', allow_pickle=True)[str(idx)].item()
    n = len(f['t'])
    print(f'=== {label} ({n} steps @50Hz) ===')
    for side, name in [(0, 'L'), (1, 'R')]:
        c = f['contact'][:, side].astype(bool)
        b = bouts(c)
        gaps = [b[i + 1][0] - (b[i][0] + b[i][1]) for i in range(len(b) - 1)]
        lens = [l for _, l in b]
        foot = f['footz'][:, side]
        foot = foot - foot.min()
        lifts = []
        for s, l in b:
            lo = max(0, s - 20)
            lifts.append(foot[s:s + l].max() - foot[lo:s + l].min())
        print(f' {name}: bouts={len(b)}  bout_len med={np.median(lens):.0f} steps  gap med={np.median(gaps) if gaps else None}'
              f'  lift/bout cm: med={100 * np.median(lifts):.2f} p90={100 * np.percentile(lifts, 90):.2f} max={100 * max(lifts):.2f}')
        k = f['joint'][:, [3, 9][side]]
        print(f'    knee mean={k.mean():.3f} range={k.max() - k.min():.3f} rad   footz total range={foot.max():.3f} m')
    ph = np.sin(2 * np.pi * f['t'] / 1.0)
    for side, name in [(0, 'L'), (1, 'R')]:
        k = f['joint'][:, [3, 9][side]]
        k = k - k.mean()
        corr = float((k * ph).mean() / (k.std() * ph.std() + 1e-12))
        print(f'    knee-{name} corr with sin(2*pi*t/1.0): {corr:+.3f}')
    print()
