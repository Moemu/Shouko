"""Parse the Yumi VRM skeleton: humanoid bone rest positions in meters.

VRM is a GLB container; the JSON chunk holds nodes and the VRM humanoid bone
mapping. VRM0 models face +Z after the loader's rotateVRM0, but the raw file
stores the -Z-facing pose; only distances and heights are used here, so the
facing convention does not change the measurements.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_glb_json(path):
    data = Path(path).read_bytes()
    magic, version, _length = struct.unpack_from('<4sII', data, 0)
    if magic != b'glTF':
        raise ValueError(f'{path} is not a GLB file')
    chunk_length, chunk_type = struct.unpack_from('<II', data, 12)
    if chunk_type != 0x4E4F534A:  # 'JSON'
        raise ValueError('First GLB chunk is not JSON')
    return json.loads(data[20:20 + chunk_length])


def world_transforms(gltf):
    nodes = gltf['nodes']
    world = {}

    def compose(node, parent):
        if 'matrix' in node:
            local = np.array(node['matrix'], dtype=float).reshape(4, 4).T
        else:
            t = node.get('translation', [0, 0, 0])
            r = node.get('rotation', [0, 0, 0, 1])
            s = node.get('scale', [1, 1, 1])
            x, y, z, w = r
            rot = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            local = np.eye(4)
            local[:3, :3] = rot * np.asarray(s)
            local[:3, 3] = t
        current = parent @ local
        return current

    def walk(index, parent):
        current = compose(nodes[index], parent)
        world[index] = current
        for child in nodes[index].get('children', []):
            walk(child, current)

    scene = gltf['scenes'][gltf.get('scene', 0)]
    for root in scene['nodes']:
        walk(root, np.eye(4))
    return world


def humanoid_bones(gltf):
    ext = gltf.get('extensions', {})
    if 'VRM' in ext:  # VRM 0.x
        mapping = ext['VRM']['humanoid']['humanBones']
        return {entry['bone']: entry['node'] for entry in mapping}, '0'
    if 'VRMC_vrm' in ext:  # VRM 1.0
        mapping = ext['VRMC_vrm']['humanoid']['humanBones']
        return {name: entry['node'] for name, entry in mapping.items()}, '1'
    raise ValueError('No VRM humanoid extension found')


def measure(path=ROOT / 'web/public/yumi.vrm'):
    gltf = read_glb_json(path)
    bones, version = humanoid_bones(gltf)
    world = world_transforms(gltf)
    position = {name: world[node][:3, 3] for name, node in bones.items() if node in world}

    def p(name):
        if name not in position:
            raise KeyError(f'Missing humanoid bone: {name}')
        return position[name]

    hips = p('hips')
    left = {k: p(f'left{k}') for k in ['UpperLeg', 'LowerLeg', 'Foot', 'Toes'] if f'left{k}' in position}
    right = {k: p(f'right{k}') for k in ['UpperLeg', 'LowerLeg', 'Foot', 'Toes'] if f'right{k}' in position}

    def seg(a, b):
        return float(np.linalg.norm(a - b))

    result = {
        'source': str(path),
        'vrm_version': version,
        'bones': {name: [round(float(v), 5) for v in pos] for name, pos in sorted(position.items())},
        'hips_height': round(float(hips[1]), 5),
        'hip_half_width': round(abs(float(p('leftUpperLeg')[0] - p('rightUpperLeg')[0])) / 2, 5),
        'left_thigh': round(seg(p('leftUpperLeg'), p('leftLowerLeg')), 5),
        'left_shin': round(seg(p('leftLowerLeg'), p('leftFoot')), 5),
        'right_thigh': round(seg(p('rightUpperLeg'), p('rightLowerLeg')), 5),
        'right_shin': round(seg(p('rightLowerLeg'), p('rightFoot')), 5),
        'ankle_height': round(float((p('leftFoot')[1] + p('rightFoot')[1]) / 2), 5),
        'spine_to_head': round(seg(p('spine'), p('head')) if 'spine' in position and 'head' in position else 0, 5),
        'head_height': round(float(p('head')[1]) if 'head' in position else 0, 5),
        'shoulder_width': round(abs(float(p('leftUpperArm')[0] - p('rightUpperArm')[0]))
                                if 'leftUpperArm' in position and 'rightUpperArm' in position else 0, 5),
        'arm_length': round((seg(p('leftUpperArm'), p('leftLowerArm')) + seg(p('leftLowerArm'), p('leftHand')))
                            if all(f'left{k}' in position for k in ['UpperArm', 'LowerArm', 'Hand']) else 0, 5),
        'foot_to_toe': round(seg(p('leftFoot'), p('leftToes')) if 'leftToes' in position else 0, 5),
    }
    leg = result['left_thigh'] + result['left_shin'] + result['ankle_height']
    result['leg_total'] = round(leg, 5)
    result['hips_minus_leg'] = round(result['hips_height'] - leg, 5)
    return result


if __name__ == '__main__':
    out = measure(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'web/public/yumi.vrm')
    print(json.dumps(out, indent=2, ensure_ascii=False))
