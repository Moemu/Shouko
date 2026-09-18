"""Numerical visual-consistency check: VRM front-end FK vs MuJoCo physics pose.

Replicates web/main.js updateBody() math offline:
  - avatarRoot.rotation.y = pi (VRMUtils.rotateVRM0), avatarAxis = -1 (VRM0)
  - hips.quaternion.set(q5*aa, q6, q4*aa, q3)
  - rotateBone: local quat = qX(pitch*aa) * qZ(roll*aa) * qY(yaw)
  - avatarRoot.position = (q1*s, q2*s - hipsRest, q0*s), s = hipsRest/physicsHips
Compares the resulting VRM ankle/knee/sole world positions against MuJoCo
xpos/geom_xpos. Read-only: writes nothing, touches no server.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'app'))

import mujoco  # noqa: E402
from sim import Body  # noqa: E402
from yumi_skeleton import read_glb_json, humanoid_bones, world_transforms  # noqa: E402


# ---------- quaternion helpers, three.js convention (x, y, z, w) ----------
def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz])


def qrot(q, v):
    qv = np.array([v[0], v[1], v[2], 0.0])
    qc = np.array([-q[0], -q[1], -q[2], q[3]])
    return qmul(qmul(q, qv), qc)[:3]


def qaxis(axis, angle):
    h = angle / 2.0
    return np.array([axis[0] * np.sin(h), axis[1] * np.sin(h), axis[2] * np.sin(h), np.cos(h)])


X, Y, Z = np.array([1., 0, 0]), np.array([0., 1, 0]), np.array([0., 0, 1])
ROT_Y_PI = np.array([0., 1., 0., 0.])  # rotateVRM0: scene.rotation.y = pi

# ---------- VRM rest skeleton ----------
gltf = read_glb_json(ROOT / 'web/public/yumi.vrm')
bones, version = humanoid_bones(gltf)
world = world_transforms(gltf)
raw = {name: world[node][:3, 3].astype(float) for name, node in bones.items()}
assert version == '0'
AA = -1.0  # avatarAxis for VRM0
# normalized-rig rest positions after rotateVRM0 (negate x and z)
norm = {k: qrot(ROT_Y_PI, v) for k, v in raw.items()}
hips_rest = float(norm['hips'][1])
physics_hips = 0.97231  # meta.body.hips_height served to the front end
SCALE = hips_rest / physics_hips


def frontend_fk(qpos):
    """Mirror of updateBody(): returns display-frame world positions."""
    q = qpos
    t_root = np.array([q[1] * SCALE, q[2] * SCALE - hips_rest, q[0] * SCALE])
    hips_pos = t_root + qrot(ROT_Y_PI, norm['hips'])
    qh = np.array([q[5] * AA, q[6], q[4] * AA, q[3]])
    qh /= np.linalg.norm(qh)
    r_hips = qmul(ROT_Y_PI, qh)

    def local(pitch=0.0, roll=0.0, yaw=0.0):
        return qmul(qmul(qaxis(X, pitch * AA), qaxis(Z, roll * AA)), qaxis(Y, yaw))

    out = {}
    for side, i in (('left', 7), ('right', 13)):
        p, r, y, knee, ap, ar = q[i:i + 6]
        up, lo, ft = (raw[f'{side}UpperLeg'], raw[f'{side}LowerLeg'], raw[f'{side}Foot'])
        r_u = qmul(r_hips, local(p, r, y))
        hip_pos = hips_pos + qrot(r_hips, up - raw['hips'])
        knee_pos = hip_pos + qrot(r_u, lo - up)
        r_l = qmul(r_u, local(knee))
        ankle_pos = knee_pos + qrot(r_l, ft - lo)
        r_f = qmul(r_l, local(ap, ar))
        # sole point: directly below the ankle by rest ankle height, in foot frame
        sole_off = np.array([0.0, -raw[f'{side}Foot'][1], 0.0])
        sole_pos = ankle_pos + qrot(r_f, sole_off)
        out[side] = dict(hips=hips_pos, hip=hip_pos, knee=knee_pos, ankle=ankle_pos,
                         sole=sole_pos, r_foot=r_f)
    return out


def disp2phys(p):
    """display (x,y,z) -> physics (x_fwd, y_left, z_up), inverse of main.js mapping."""
    return np.array([p[2] / SCALE, p[0] / SCALE, p[1] / SCALE])


# ---------- MuJoCo ground truth ----------
body = Body(load_motor_policy=False, robot='yumi')
model, data = body.model, body.data
bid = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'{s}_ankle_roll_link') for s in ('left', 'right')}
kid = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'{s}_knee_link') for s in ('left', 'right')}
foot_geoms = [g for g in range(model.ngeom)
              if model.geom_bodyid[g] in bid.values() and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]


def mujoco_points(side):
    ankle = data.xpos[bid[side]].copy()
    knee = data.xpos[kid[side]].copy()
    soles = [data.geom_xpos[g][2] - model.geom_size[g][0] for g in foot_geoms
             if model.geom_bodyid[g] == bid[side]]
    return ankle, knee, min(soles)


def quat_from_mat(m):
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        return np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                         (m[1, 0] - m[0, 1]) / s, s / 4])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(m[i, i] - m[j, j] - m[k, k] + 1) * 2
    q = np.zeros(4)
    q[i] = s / 4
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    q[3] = (m[k, j] - m[j, k]) / s
    return q


def qangle(a, b):
    d = abs(float(np.dot(a, b)))
    return 2 * np.degrees(np.arccos(min(d, 1.0)))


# physics (x_fwd,y_left,z_up) -> display (x,y,z) permutation: d = (y, z, x)
PERM = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)


def compare(tag):
    fk = frontend_fk(data.qpos)
    rows = {}
    for side in ('left', 'right'):
        ankle_mj, knee_mj, sole_mj = mujoco_points(side)
        hip_mj = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                             f'{side}_hip_pitch_link')].copy()
        ankle_fk = disp2phys(fk[side]['ankle'])
        knee_fk = disp2phys(fk[side]['knee'])
        hip_fk = disp2phys(fk[side]['hip'])
        sole_fk_z = fk[side]['sole'][1] / SCALE
        # foot orientation: MuJoCo ankle_roll_link xmat -> display frame
        r_mj = data.xmat[bid[side]].reshape(3, 3)
        r_mj_disp = quat_from_mat(PERM @ r_mj @ PERM.T)
        # strip the constant scene flip (rotateVRM0) before comparing
        r_fk_model = qmul(fk[side]['r_foot'], np.array([0., -1., 0., 0.]))
        foot_rot_err = qangle(r_fk_model, r_mj_disp)
        rows[side] = dict(
            ankle_err=float(np.linalg.norm(ankle_fk - ankle_mj)),
            ankle_fk=ankle_fk, ankle_mj=ankle_mj,
            knee_err=float(np.linalg.norm(knee_fk - knee_mj)),
            thigh_vec_err=float(np.linalg.norm((knee_fk - hip_fk) - (knee_mj - hip_mj))),
            shin_vec_err=float(np.linalg.norm((ankle_fk - knee_fk) - (ankle_mj - knee_mj))),
            foot_rot_err=float(foot_rot_err),
            knee_fwd_mj=float(knee_mj[0] - ankle_mj[0]),
            knee_fwd_fk=float(knee_fk[0] - ankle_fk[0]),
            sole_fk_z=float(sole_fk_z), sole_mj_z=float(sole_mj))
    return rows


# ===== test 1: PD hold at home pose; sample trajectory then settle =====
samples = {}
stand = None
for n in range(500):  # 10 s
    state = body.step_joints(np.zeros(12))
    if n + 1 == 50:
        stand = compare('stand-1s')  # last upright instant (PD hold later falls)
    if n + 1 in (50, 100, 250, 500):
        samples[n + 1] = (state['height'], state['upright'], state['fallen'])
for k, v in samples.items():
    print(f'[stand] t={k*0.02:.1f}s pelvis_z={v[0]:.4f} upright={v[1]:.3f} fallen={v[2]}')
for side, r in stand.items():
    print(f'  {side}: ankle FK err {r["ankle_err"]*100:.2f} cm | knee FK err {r["knee_err"]*100:.2f} cm')
    print(f'        ankle mj {np.round(r["ankle_mj"],4)} fk {np.round(r["ankle_fk"],4)}')
    print(f'        sole z: mj {r["sole_mj_z"]:.4f}  fk {r["sole_fk_z"]:.4f}  '
          f'knee-ankle fwd: mj {r["knee_fwd_mj"]:.4f} fk {r["knee_fwd_fk"]:.4f}')

# ===== test 2: randomized poses, mapping fidelity =====
# rest-pose (zero) offsets per leg, to separate structural geometry vs rotation mapping
mujoco.mj_resetData(model, data)
data.qpos[2] = 0.97231
mujoco.mj_forward(model, data)
zero_rows = compare('zero-ref')
zero_off = {s: zero_rows[s]['ankle_fk'] - zero_rows[s]['ankle_mj'] for s in ('left', 'right')}
for s in ('left', 'right'):
    a_m, a_f = zero_rows[s]['ankle_mj'], zero_rows[s]['ankle_fk']
    print(f'[zero-pose decomp] {s}: ankle mj {np.round(a_m,4)} fk {np.round(a_f,4)} '
          f'offset {np.round(zero_off[s],4)}')
    k_m = data.xpos[kid[s]].copy()
    print(f'          knee mj {np.round(k_m,4)} vs hip->knee check')

rng = np.random.default_rng(7)
jr = model.jnt_range[1:13]  # 12 hinge joints (free joint is jnt 0)
worst = 0.0
worst_rel = 0.0
worst_vec = 0.0
worst_rot = 0.0
knee_sign_bad = 0
for n in range(300):
    mujoco.mj_resetData(model, data)
    data.qpos[0:2] = rng.normal(0, 0.3, 2)
    data.qpos[2] = rng.uniform(0.6, 1.1)
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    ang = rng.uniform(0, 0.6)
    data.qpos[3:7] = [np.cos(ang / 2), *(np.sin(ang / 2) * ax)]
    for j in range(12):
        lo, hi = jr[j]
        span = min(hi - lo, 1.2)
        mid = (hi + lo) / 2
        data.qpos[7 + j] = np.clip(mid + rng.uniform(-span / 2, span / 2), lo, hi)
    mujoco.mj_forward(model, data)
    rows = compare(f'rand{n}')
    for side, r in rows.items():
        worst = max(worst, r['ankle_err'], r['knee_err'])
        rel = np.linalg.norm((r['ankle_fk'] - r['ankle_mj']) - zero_off[side])
        worst_rel = max(worst_rel, float(rel))
        worst_vec = max(worst_vec, r['thigh_vec_err'], r['shin_vec_err'])
        worst_rot = max(worst_rot, r['foot_rot_err'])
        if abs(r['knee_fwd_mj']) > 0.03 and np.sign(r['knee_fwd_mj']) != np.sign(r['knee_fwd_fk']):
            knee_sign_bad += 1
print(f'[random] 300 poses x 2 legs: worst ankle/knee FK err {worst*100:.2f} cm, '
      f'worst err after removing rest offset {worst_rel*100:.2f} cm,\n'
      f'         worst thigh/shin segment-vector err {worst_vec*100:.3f} cm, '
      f'worst foot rotation err {worst_rot:.3f} deg,\n'
      f'         knee-direction sign mismatches: {knee_sign_bad}')

# ===== test 3: zero pose (straight legs) FK sanity =====
mujoco.mj_resetData(model, data)
data.qpos[2] = 0.97231
mujoco.mj_forward(model, data)
rows = compare('zero')
for side, r in rows.items():
    print(f'[zero] {side}: ankle err {r["ankle_err"]*100:.2f} cm, knee err {r["knee_err"]*100:.2f} cm, '
          f'sole z mj {r["sole_mj_z"]:.4f} fk {r["sole_fk_z"]:.4f}')

# ===== test 4: kinematic home stance (the physics standing pose) =====
mujoco.mj_resetData(model, data)
data.qpos[7:] = body.home
data.qpos[2] = body.initial_height
mujoco.mj_forward(model, data)
rows = compare('home')
for side, r in rows.items():
    print(f'[home] {side}: ankle err {r["ankle_err"]*100:.2f} cm, sole z mj {r["sole_mj_z"]:.4f} '
          f'fk {r["sole_fk_z"]:.4f}, knee-ankle fwd mj {r["knee_fwd_mj"]:.4f} fk {r["knee_fwd_fk"]:.4f}')
