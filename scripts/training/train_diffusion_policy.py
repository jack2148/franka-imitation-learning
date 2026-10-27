#!/usr/bin/env python3
if __package__:
    from .train_lerobot_policy import main_for_policy
else:
    from train_lerobot_policy import main_for_policy

if __name__ == "__main__":
    raise SystemExit(main_for_policy("diffusion"))
