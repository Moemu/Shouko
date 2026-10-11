"""Isolated 18-joint Yumi proxy for goal-task experiments, without a motor policy."""
from pathlib import Path
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
LEG_JOINTS = tuple(f'{side}_{part}_joint' for side in ('left', 'right')
                   for part in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll'))
ARM_JOINTS = tuple(f'{side}_{part}_joint' for side in ('left', 'right')
                   for part in ('shoulder_pitch', 'shoulder_roll', 'elbow'))
JOINT_NAMES = LEG_JOINTS + ARM_JOINTS
UPPER_LENGTH, LOWER_LENGTH = 0.25196, 0.22701
SHOULDER_HEIGHT, SHOULDER_HALF_WIDTH = 0.36355, 0.18


@dataclass
class ContactSummary:
    count: int = 0
    constraints: int = 0
    penetration: float = 0.
    limit_error: float = 0.
    ground_bodies: set[str] = field(default_factory=set)


def parallel_axis(position: np.ndarray) -> np.ndarray:
    return np.dot(position, position) * np.eye(3) - np.outer(position, position)


def model_xml() -> str:
    """Build from the released leg geometry; subtract the hanging arms' rigid inertia."""
    root = ET.parse(ROOT / 'app/yumi_description/yumi.xml').getroot()
    root.set('model', 'yumi_free_18dof')
    ET.SubElement(root, 'option', timestep='0.002', iterations='50', ls_iterations='50', tolerance='1e-6')
    world = root.find('worldbody')
    pelvis = world.find("body[@name='pelvis']")
    actuator = root.find('actuator')
    contact = ET.SubElement(root, 'contact')
    # The inherited torso and proximal thigh capsules overlap at the hip attachment.
    for side in ('left', 'right'):
        ET.SubElement(contact, 'exclude', body1='pelvis', body2=f'{side}_hip_yaw_link')
    arm_parts: list[tuple[float, np.ndarray, np.ndarray]] = []

    def inertial(body: ET.Element, mass: float, center: np.ndarray, diagonal: np.ndarray,
                 world_center: np.ndarray) -> None:
        ET.SubElement(body, 'inertial', mass=str(mass), pos=' '.join(map(str, center)),
                      diaginertia=' '.join(map(str, diagonal)))
        arm_parts.append((mass, world_center, np.diag(diagonal)))

    for side, sign in (('left', 1), ('right', -1)):
        shoulder = np.array([0., sign * SHOULDER_HALF_WIDTH, SHOULDER_HEIGHT])
        pitch = ET.SubElement(pelvis, 'body', name=f'{side}_shoulder_pitch_link',
                              pos=' '.join(map(str, shoulder)))
        inertial(pitch, .02, np.zeros(3), np.full(3, 1e-5), shoulder)
        ET.SubElement(pitch, 'joint', name=f'{side}_shoulder_pitch_joint', axis='0 1 0',
                      range='-2.5 2.5', actuatorfrcrange='-25 25')
        upper = ET.SubElement(pitch, 'body', name=f'{side}_shoulder_roll_link')
        ET.SubElement(upper, 'joint', name=f'{side}_shoulder_roll_joint', axis='1 0 0',
                      range='-1.8 1.8', actuatorfrcrange='-25 25')
        fore = ET.SubElement(upper, 'body', name=f'{side}_elbow_link', pos=f'0 0 {-UPPER_LENGTH}')
        ET.SubElement(fore, 'joint', name=f'{side}_elbow_joint', axis='0 1 0',
                      range='0 2.6', actuatorfrcrange='-15 15')
        for body, mass, length, radius, origin in (
                (upper, .60, UPPER_LENGTH, .035, shoulder),
                (fore, .35, LOWER_LENGTH, .03, shoulder + [0, 0, -UPPER_LENGTH])):
            center = np.array([0., 0., -length / 2])
            diagonal = np.array([mass * (3 * radius**2 + length**2) / 12] * 2
                                + [mass * radius**2 / 2])
            inertial(body, mass, center, diagonal, origin + center)
            ET.SubElement(body, 'geom', type='capsule', size=str(radius),
                          fromto=f'0 0 0 0 0 {-length}', density='0', rgba='.3 .65 .8 1')
        ET.SubElement(fore, 'geom', name=f'{side}_palm', type='sphere', size='.04',
                      pos=f'0 0 {-LOWER_LENGTH}', density='0', rgba='.2 .4 .6 1')
        for part in ('shoulder_pitch', 'shoulder_roll', 'elbow'):
            name = f'{side}_{part}_joint'
            ET.SubElement(actuator, 'motor', name=name, joint=name)

    old = pelvis.find('inertial')
    old_mass = float(old.get('mass'))
    old_center = np.fromstring(old.get('pos'), sep=' ')
    old_inertia = np.diag(np.fromstring(old.get('diaginertia'), sep=' '))
    mass = old_mass - sum(part[0] for part in arm_parts)
    center = (old_mass * old_center - sum(m * p for m, p, _ in arm_parts)) / mass
    inertia = old_inertia + old_mass * parallel_axis(old_center)
    inertia -= sum(i + m * parallel_axis(p) for m, p, i in arm_parts)
    inertia -= mass * parallel_axis(center)
    eigenvalues = np.linalg.eigvalsh(inertia)
    if eigenvalues[0] <= 0 or eigenvalues[-1] > eigenvalues[:2].sum():
        raise ValueError('Arm mass split produces an invalid torso inertia')
    old.attrib.clear()
    old.set('mass', str(mass))
    old.set('pos', ' '.join(map(str, center)))
    old.set('fullinertia', ' '.join(map(str, [inertia[0, 0], inertia[1, 1], inertia[2, 2],
                                            inertia[0, 1], inertia[0, 2], inertia[1, 2]])))
    for index, geom in enumerate(root.iter('geom')):
        geom.set('contype', '1')
        geom.set('conaffinity', '1')
        geom.set('solref', '.008 1')
        if 'name' not in geom.attrib:
            geom.set('name', f'body_geom_{index}')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='0 0 .05', rgba='.2 .3 .4 1', solref='.008 1')
    return ET.tostring(root, encoding='unicode')


class FreeBody:
    def __init__(self) -> None:
        self.xml = model_xml()
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.pelvis = self.model.body('pelvis').id
        self.floor = self.model.geom('floor').id
        self.joints = np.array([self.model.joint(name).id for name in JOINT_NAMES])
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.vadr = self.model.jnt_dofadr[self.joints]
        self.actuators = np.array([self.model.actuator(name).id for name in JOINT_NAMES])
        cfg = yaml.safe_load((ROOT / 'app/yumi_description/yumi.yaml').read_text(encoding='utf-8'))
        self.home = np.array(cfg['default_angles'] + [0.] * 6)
        self.kp = np.array(cfg['kps'] + [40., 40., 30.] * 2)
        self.kd = np.array(cfg['kds'] + [2., 2., 1.5] * 2)
        self.ranges = self.model.jnt_range[self.joints]
        self.limits = self.model.jnt_actfrcrange[self.joints, 1]
        self.decimation = int(cfg['control_decimation'])
        self.action_scale = .25
        self.dt = self.model.opt.timestep * self.decimation
        self.action = np.zeros(18)
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = .962
        self.data.qpos[self.qadr] = self.home
        self.action.fill(0)
        self.work = 0.
        self.last_contacts = ContactSummary()
        self.refresh()

    def refresh(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_subtreeVel(self.model, self.data)

    @property
    def center(self) -> np.ndarray:
        return self.data.subtree_com[self.pelvis].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.data.subtree_linvel[self.pelvis].copy()

    @property
    def rotation(self) -> np.ndarray:
        return self.data.xmat[self.pelvis].reshape(3, 3)

    def step(self, action: np.ndarray) -> float:
        action = np.asarray(action, dtype=float)
        if action.shape != (18,) or not np.isfinite(action).all():
            raise ValueError('Expected 18 finite joint actions')
        self.action = np.clip(action, -8, 8)
        target = np.clip(self.home + self.action_scale * self.action, self.ranges[:, 0], self.ranges[:, 1])
        work = 0.
        self.last_contacts = ContactSummary()
        for _ in range(self.decimation):
            torque = self.kp * (target - self.data.qpos[self.qadr]) - self.kd * self.data.qvel[self.vadr]
            self.data.ctrl[self.actuators] = np.clip(torque, -self.limits, self.limits)
            # mj_step1/2 exposes applied joint force and the matching pre-integration velocity.
            mujoco.mj_step1(self.model, self.data)
            contact = self.contacts()
            self.last_contacts.count = max(self.last_contacts.count, contact.count)
            self.last_contacts.constraints = max(self.last_contacts.constraints, contact.constraints)
            self.last_contacts.penetration = max(self.last_contacts.penetration, contact.penetration)
            self.last_contacts.ground_bodies.update(contact.ground_bodies)
            self.last_contacts.limit_error = max(self.last_contacts.limit_error, contact.limit_error)
            mujoco.mj_fwdActuation(self.model, self.data)
            work += float(np.abs(self.data.qfrc_actuator[self.vadr] * self.data.qvel[self.vadr]).sum()) * self.model.opt.timestep
            mujoco.mj_step2(self.model, self.data)
        self.work += work
        self.refresh()
        return work

    def contacts(self) -> ContactSummary:
        ground_bodies = set()
        for contact in self.data.contact:
            if self.floor in (contact.geom1, contact.geom2):
                other = contact.geom2 if contact.geom1 == self.floor else contact.geom1
                ground_bodies.add(self.model.body(self.model.geom_bodyid[other]).name)
        q = self.data.qpos[self.qadr]
        return ContactSummary(count=int(self.data.ncon), constraints=int(self.data.nefc),
                              penetration=max((max(0., -float(c.dist)) for c in self.data.contact), default=0.),
                              limit_error=max(0., float(np.max(np.maximum(self.ranges[:, 0] - q,
                                                                          q - self.ranges[:, 1])))),
                              ground_bodies=ground_bodies)
