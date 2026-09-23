"""Explicit heading-start coverage for bounded PPO recovery experiments."""
import torch


class RecoveryCurriculum:
    def __init__(self, env, yaws, seconds):
        self.env = env
        self.yaws = torch.tensor(yaws, device=env.qpos.device, dtype=env.qpos.dtype)
        self.limit = round(seconds/env.dt) if seconds > 0 else 0
        self.starts = torch.zeros((env.worlds, len(yaws)), dtype=torch.long, device=env.qpos.device)
        self.next_yaw = torch.arange(env.worlds, device=env.qpos.device) % len(yaws)

    def apply(self, mask):
        indices = mask.nonzero(as_tuple=True)[0]
        selection = self.next_yaw[indices]
        yaw = self.yaws[selection]
        self.env.qpos[indices, 3] = torch.cos(yaw/2)
        self.env.qpos[indices, 4:6] = 0
        self.env.qpos[indices, 6] = torch.sin(yaw/2)
        self.starts[indices, selection] += 1
        self.next_yaw[indices] = (selection+1) % len(self.yaws)

    def truncated(self, fallen):
        return ((self.env.episode_steps >= self.limit) & ~fallen if self.limit
                else torch.zeros_like(fallen))

    def summary(self):
        return dict(yaws=self.yaws.cpu().tolist(), starts_by_world=self.starts.cpu().tolist(),
                    episode_limit_steps=self.limit)


def bootstrap_timeouts(reward, final_value, truncated, gamma):
    """Move the final-state bootstrap into reward before marking a reset boundary."""
    return reward + gamma * final_value * truncated.to(reward.dtype)
