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

    # >>> 训练期奖励形塑开关与系数 <<<
    shaping_enable: bool = False   # 对应 train_ppo 的 --shaping.enable
    shaping_beta: float = 1.0      # 对应 --shaping.beta
    shaping_eta: float = 0.0       # 对应 --shaping.eta
    shaping_alpha: float = 0.5     # 对应 --shaping.alpha

    # >>> Phase-1：信念推进参数（透传到 train_ppo） <<<
    belief_enable: bool = False
    belief_ema_tau: float = 0.9
    belief_temperature: float = 1.0
    belief_reward_coef_delta: float = 0.0
    belief_expose_accept_prob: bool = False

    # >>> 命题噪声参数（透传到 train_ppo） <<<
    noise_enable: bool = False
    noise_p_miss: float = 0.0
    noise_p_false: float = 0.0
    noise_delay_steps: int = 0


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

        # 形塑参数（仅启用时透传）
        if args.shaping_enable:
            command += [
                '--shaping.enable',
                '--shaping.beta', str(args.shaping_beta),
                '--shaping.eta',  str(args.shaping_eta),
                '--shaping.alpha', str(args.shaping_alpha),
            ]

        # Phase-1：信念推进（仅启用时透传）
        if args.belief_enable:
            command += [
                '--belief.enable',
                '--belief.ema_tau', str(args.belief_ema_tau),
                '--belief.temperature', str(args.belief_temperature),
                '--belief.reward_coef_delta', str(args.belief_reward_coef_delta),
            ]
            if args.belief_expose_accept_prob:
                command += ['--belief.expose_accept_prob']

        # 命题噪声（仅启用时透传）
        if args.noise_enable:
            command += [
                '--noise_enable',
                '--noise_p_miss', str(args.noise_p_miss),
                '--noise_p_false', str(args.noise_p_false),
                '--noise_delay_steps', str(args.noise_delay_steps),
            ]

        if args.log_wandb:
            command.append('--log_wandb')
        if not args.log_csv:
            command.append('--no-log_csv')
        if not args.save:
            command.append('--no-save')

        subprocess.run(command, env=env)


if __name__ == '__main__':
    if len(sys.argv) == 1:  # defaults
        sys.argv += '--num_procs 2 --device cuda --name belief_test --seed 1 --log_csv true --save true'.split(' ')
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted!')
        wandb.finish()
        # kill_all_wandb_processes()
        sys.exit(0)

# 示例：
# python run_zones.py \
#   --name ppo_belief_eta002 \
#   --seed 1 \
#   --device cuda \
#   --num_procs 16 \
#   --log_csv true --save true \
#   --shaping_enable true --shaping_beta 0.01 --shaping_eta 0.02 --shaping_alpha 0.5 \
#   --belief_enable true --belief_ema_tau 0.9 --belief_temperature 1.0 \
#   --belief_reward_coef_delta 0.02 --belief_expose_accept_prob true \
#   --noise_enable true --noise_p_miss 0.05 --noise_p_false 0.02 --noise_delay_steps 1
