"""Training-only demonstrator. Its recurrent memory belongs to each physical world."""
import torch

from .gpu_body import UNITREE


def load_teacher(worlds):
    teacher = torch.jit.load(str(UNITREE/'deploy/pre_train/g1/motion.pt'), map_location='cuda').eval()
    teacher.hidden_state = torch.zeros(1, worlds, 64, device='cuda')
    teacher.cell_state = torch.zeros(1, worlds, 64, device='cuda')
    return teacher


def reset_teacher(teacher, mask):
    teacher.hidden_state[:, mask] = 0
    teacher.cell_state[:, mask] = 0
