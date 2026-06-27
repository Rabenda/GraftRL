#!/usr/bin/env python3
"""Dry-run Sokoban env + action parser (no GPU / no verl rollout)."""

from __future__ import annotations

import os
import sys

VERL_AGENT_ROOT = os.environ.get("VERL_AGENT_ROOT", "/workspace/repo/verl-agent")
if VERL_AGENT_ROOT not in sys.path:
    sys.path.insert(0, VERL_AGENT_ROOT)

from examples.profile.workloads.sokoban.sokoban_agent_loop import _parse_action, _rgb_to_pil  # noqa: E402
from agent_system.environments.env_package.sokoban.sokoban.env import SokobanEnv  # noqa: E402


def main() -> None:
    env = SokobanEnv(
        mode="rgb_array",
        dim_room=(6, 6),
        num_boxes=1,
        max_steps=15,
        search_depth=30,
    )
    obs, _ = env.reset(seed=0)
    images = [_rgb_to_pil(obs)]
    print(f"reset: obs={obs.shape if hasattr(obs, 'shape') else type(obs)} images={len(images)}")

    for step in range(1, 6):
        action = (step % 4) + 1
        obs, _reward, terminated, _info = env.step(action)
        images.append(_rgb_to_pil(obs))
        print(
            f"step={step} action={action} term={terminated} success={env.success()} "
            f"accumulated_images={len(images)}"
        )
        if terminated or env.success():
            break

    text = "plan<action>right</action>"
    print(f"parse_action({text!r}) -> {_parse_action(text)}")
    env.close()
    print("smoke OK")


if __name__ == "__main__":
    main()
