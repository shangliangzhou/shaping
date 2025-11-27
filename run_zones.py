#!/usr/bin/env python
import os
import subprocess
import sys
from dataclasses import dataclass
import simple_parsing
import wandb

from src.utils.utils import kill_all_wandb_processes  # 若不需要可移除


@dataclass
class Args:
    name: str
    seed: int | list[int]
    device: str
    num_procs: int = 16
    log_csv: bool = True
    log_wandb: bool = False
    save: bool = True

    # ===== 信念化（BeliefWrapper）参数：run 时透传到 train_ppo.py =====
    belief_enable: bool = True
    belief_theta_up: float = 0.8
    belief_theta_down: float = 0.2
    belief_k_stab: int = 3
    belief_ema_lambda: float = 0.8
    belief_k: float = 5.0
    belief_m: float = 0.0


def main():
    args = simple_parsing.parse(Args)
    env = os.environ.copy()
    env['PYTHONPATH'] = 'src/'
    seeds = args.seed if isinstance(args.seed, list) else [args.seed]

    for seed in seeds:
        command = [
            'python', 'src/train/train_ppo.py',
            '--env', 'PointLtl2-v0',
            '--steps_per_process', '4096',
            '--batch_size', '2048',
            '--lr', '0.0003',
            '--discount', '0.998',
            '--entropy_coef', '0.003',
            '--log_interval', '1',
            '--save_interval', '2',
            '--epochs', '10',
            '--num_steps', '15_000_000',
            '--model_config', 'PointLtl2-v0',
            '--curriculum', 'PointLtl2-v0',
            '--name', args.name,
            '--seed', str(seed),
            '--device', args.device,
            '--num_procs', str(args.num_procs),
        ]

        # ===== 透传信念化参数 =====
        if args.belief_enable:
            command += ['--belief.enable']
        else:
            command += ['--no-belief.enable']

        command += [
            '--belief.theta_up', str(args.belief_theta_up),
            '--belief.theta_down', str(args.belief_theta_down),
            '--belief.k_stab', str(args.belief_k_stab),
            '--belief.ema_lambda', str(args.belief_ema_lambda),
            '--belief.k', str(args.belief_k),
            '--belief.m', str(args.belief_m),
        ]

        if args.log_wandb:
            command.append('--log_wandb')
        if not args.log_csv:
            command.append('--no-log_csv')
        if not args.save:
            command.append('--no-save')

        subprocess.run(command, env=env)


if __name__ == '__main__':
    if len(sys.argv) == 1:  # no args -> quick default
        sys.argv += '--num_procs 2 --device cuda --name brief_belief_s1 --seed 1 --log_csv true --save true'.split(' ')
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted!')
        wandb.finish()
        # kill_all_wandb_processes()
        sys.exit(0)

# 示例：
#   python run_zones.py \
#     --name brief_belief_s1 \
#     --seed 1 \
#     --device cuda:0 \
#     --num_procs 16 \
#     --log_csv true --save true \
#     --belief_enable true --belief_theta_up 0.8 --belief_theta_down 0.2 --belief_k_stab 3
