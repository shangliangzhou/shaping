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

    # >>> 新增：训练期奖励形塑开关与系数 <<<
    shaping_enable: bool = False   # 对应 train_ppo 的 --shaping.enable
    shaping_beta: float = 1.0      # 对应 --shaping.beta
    shaping_eta: float = 0.0       # 对应 --shaping.eta

    # >>> 新增：alpha（目标匹配距离的权重） <<<
    shaping_alpha: float = 0.5

     # --- ADD: HER 相关 ---
    her_enable: bool = False
    her_k_future: int = 4
    her_aux_epochs: int = 1
    her_aux_batch: int = 1024
    her_lambda_bc: float = 0.0
    # --- END ADD ---


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
            # '--num_steps', '1500',
            '--model_config', 'PointLtl2-v0',
            '--curriculum', 'PointLtl2-v0',
            '--name', args.name,
            '--seed', str(seed),
            '--device', args.device,
            '--num_procs', str(args.num_procs),
        ]
        # >>> 仅当启用时，透传到 train_ppo.py（注意：train_ppo 用“点号”风格） <<<
        if args.shaping_enable:
            command += [
                '--shaping.enable',
                '--shaping.beta', str(args.shaping_beta),
                '--shaping.eta',  str(args.shaping_eta),
                '--shaping.alpha', str(args.shaping_alpha),  # <<< 新增透传
            ]
         # --- ADD: HER 透传 ---
        if args.her_enable:
            command += [
                '--her.enable',
                '--her.k_future', str(args.her_k_future),
                '--her.aux_epochs', str(args.her_aux_epochs),
                '--her.aux_batch', str(args.her_aux_batch),
                '--her.lambda_bc', str(args.her_lambda_bc),
            ]
        # --- END ADD ---
        if args.log_wandb:
            command.append('--log_wandb')
        if not args.log_csv:
            command.append('--no-log_csv')
        if not args.save:
            command.append('--no-save')

        subprocess.run(command, env=env)


if __name__ == '__main__':
    if len(sys.argv) == 1:  # if no arguments are provided, use the following defaults
        sys.argv += '--num_procs 2 --device cuda --name asd --seed 1 --log_csv true --save true'.split(' ')
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted!')
        wandb.finish()
        # kill_all_wandb_processes()
        sys.exit(0)

#         python run_zones.py \
#   --name zones_paper_s1 \
#   --seed 1 \
#   --device cuda:0 \
#   --num_procs 16 \
#   --log_csv true --save true
