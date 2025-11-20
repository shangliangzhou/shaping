#!/usr/bin/env python
import os
import subprocess
import sys
from dataclasses import dataclass
import simple_parsing
import wandb

from src.utils.utils import kill_all_wandb_processes


@dataclass
class Args:
    name: str
    seed: int | list[int]
    device: str
    num_procs: int = 16
    log_csv: bool = True
    log_wandb: bool = False
    save: bool = True

    # 训练期奖励形塑
    shaping_enable: bool = False     # 对应 train_ppo 的 --shaping.enable
    shaping_beta: float = 0.01       # 对应 --shaping.beta
    shaping_eta: float = 0.02        # 对应 --shaping.eta
    shaping_alpha: float = 0.5       # 对应 --shaping.alpha
    shaping_kappa: float = 0.00      # 对应 --shaping.kappa（路点密化强度）

    # Waypoint
    waypoint_enable: bool = False
    waypoint_world_scale: float = 1.0


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

        # 透传日志/保存
        if args.log_wandb:
            command.append('--log_wandb')
        if not args.log_csv:
            command.append('--no-log_csv')
        if not args.save:
            command.append('--no-save')

        # 透传 shaping（点号风格）
        if args.shaping_enable:
            command += [
                '--shaping.enable',
                '--shaping.beta', str(args.shaping_beta),
                '--shaping.eta',  str(args.shaping_eta),
                '--shaping.alpha', str(args.shaping_alpha),
                '--shaping.kappa', str(args.shaping_kappa),
            ]

        # 透传 waypoint（点号风格）
        if args.waypoint_enable:
            command += [
                '--waypoint.enable',
                '--waypoint.world_scale', str(args.waypoint_world_scale),
            ]

        subprocess.run(command, env=env)


if __name__ == '__main__':
    if len(sys.argv) == 1:  # 若无参数，给一个可跑的默认
        sys.argv += (
            '--num_procs 16 --device cuda --name ppo_wp_k02 --seed 1 '
            '--log_csv true --save true '
            '--shaping_enable true --shaping_beta 0.01 --shaping_eta 0.02 --shaping_alpha 0.5 --shaping_kappa 0.02 '
            '--waypoint_enable --waypoint_world_scale 1.0'
        ).split(' ')
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted!')
        wandb.finish()
        # kill_all_wandb_processes()
        sys.exit(0)

# 示例：
# python run_zones.py --device cuda --name ppo_wp_k02 --seed 1 \
#   --num_procs 16 --log_csv true --save true \
#   --shaping_enable true --shaping_beta 0.01 --shaping_eta 0.02 --shaping_alpha 0.5 --shaping_kappa 0.02 \
#   --waypoint_enable --waypoint_world_scale 1.0




