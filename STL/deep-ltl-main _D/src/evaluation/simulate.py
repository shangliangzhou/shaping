#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

# ====== [关键修复] 只使用“当前仓库”的 src 路径 ======
THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parents[1]          # .../src
PROJECT_ROOT = THIS_FILE.parents[2]     # 项目根
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from envs import make_env
from ltl import FixedSampler
from model.model import build_model
from model.agent import Agent
from config import model_configs
from sequence.search import ExhaustiveSearch
from utils.model_store import ModelStore

# ================= 手动配置区（无命令行也能跑） =================
# 说明：
# - 如果直接 python src/evaluation/simulate.py，不带任何参数，就用这里的配置。
# - 想要单模型：把 EXPS = ['你的实验名']，AUTO_PICK=False。
# - 想要双模型：把 EXPS = ['exp1', 'exp2']，AUTO_PICK=False。
# - 想自动选择最近的实验：把 EXPS=None，AUTO_PICK=True，并设置 AUTO_PICK_K 为 1 或 2。
ENV_NAME: str = 'PointLtl2-v0'
SEED: int = 1
NUM_EPISODES: int = 500
# FORMULA: str = '(F blue) & (!blue U (green & F yellow))'  # 有限时域示例 GF blue & GF green & GF yellow & G !magenta
FORMULA: str = 'GF blue & GF green & GF yellow & G !magenta'
FINITE: bool = False
RENDER: bool = False
DETERMINISTIC: bool = True

# 手动设定要评测的实验名（目录名），None 表示自动发现
EXPS: Optional[List[str]] = ["zones_paper_s1","base_hard_noshaping"] # 例如 ['zones_paper_s1'] 或 ['zones_paper_s1','ppo_shaping_alpha05_test']
AUTO_PICK: bool = True            # True 则自动扫描 experiments/ppo/<env>/ 下的实验
AUTO_PICK_K: int = 2              # 自动选择最近的 K 个（1=单模型，2=双模型）

# ===============================================================


def _gamma_for_env(env: str) -> float:
    if env == 'LetterEnv-v0':
        return 0.94
    elif env == 'PointLtl2-v0':
        return 0.998
    else:
        return 0.98


def _env_reset(env):
    """兼容 Gym / Gymnasium：reset 可能返回 obs 或 (obs, info)"""
    out = env.reset()
    if isinstance(out, tuple):
        if len(out) == 2:
            return out[0], out[1]
        else:
            # 兜底：只当 obs
            return out[0], {}
    return out, {}


def _env_step(env, action):
    """兼容 Gym / Gymnasium：step 返回四元/五元组"""
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, reward, done, info
    elif isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, bool(done), info
    else:
        raise RuntimeError("env.step 返回格式异常：期望4或5元组")


def _discover_experiments(env_name: str, seed: int, k: int) -> List[str]:
    """在当前仓库下，自动发现最近修改的 K 个实验目录（包含 {seed}/status.pth）。"""
    root = PROJECT_ROOT / 'experiments' / 'ppo' / env_name
    if not root.exists():
        return []

    candidates = []
    for d in root.iterdir():
        if d.is_dir():
            status = d / str(seed) / 'status.pth'
            if status.exists():
                mtime = status.stat().st_mtime
                candidates.append((mtime, d.name))
    # 按 status.pth 修改时间降序
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [name for _, name in candidates[:k]]


def _ensure_exps(env_name: str, exps: Optional[List[str]], seed: int,
                 auto_pick: bool, auto_pick_k: int) -> List[str]:
    """返回最终要评测的实验名列表，并做存在性检查。"""
    if exps is None and auto_pick:
        picked = _discover_experiments(env_name, seed, auto_pick_k)
        if not picked:
            root = PROJECT_ROOT / 'experiments' / 'ppo' / env_name
            raise FileNotFoundError(
                f"[自动扫描] 在 {root} 下未发现包含 seed={seed}/status.pth 的实验目录。"
                f"\n请先训练，或在本脚本手动设置 EXPS=['你的实验名'] 并确保存在 {root}/<exp>/{seed}/status.pth"
            )
        print(f"[自动选择] 评测实验：{picked}")
        return picked

    if not exps:
        raise ValueError("未指定 EXPS 且未启用 AUTO_PICK，无法确定要评测的实验。请在手动配置区设置 EXPS 或开启 AUTO_PICK。")

    # 存在性检查
    missing = []
    for e in exps:
        status = PROJECT_ROOT / 'experiments' / 'ppo' / env_name / e / str(seed) / 'status.pth'
        if not status.exists():
            missing.append(str(status))
    if missing:
        have = [p.name for p in (PROJECT_ROOT / 'experiments' / 'ppo' / env_name).glob('*') if p.is_dir()]
        raise FileNotFoundError(
            "[错误] 以下 status.pth 不存在：\n  - " + "\n  - ".join(missing) +
            f"\n[提示] 当前可用实验目录（不保证有对应 seed）：{have}"
        )
    return exps


def simulate(env_name: str, gamma: float, exp: str, seed: int, num_episodes: int,
             formula: str, finite: bool, render: bool, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    sampler = FixedSampler.partial(formula)
    env = make_env(env_name, sampler, render_mode='human' if render else None)
    config = model_configs[env_name]

    model_store = ModelStore(env_name, exp, seed)
    try:
        training_status = model_store.load_training_status(map_location='cpu')
    except FileNotFoundError as e:
        # 友好提示
        root = PROJECT_ROOT / 'experiments' / 'ppo' / env_name
        have = [p.name for p in root.glob("*") if p.is_dir()]
        raise FileNotFoundError(
            f"{e}\n[提示] 当前可用实验目录：{have}\n"
            f"[提示] 检查路径是否为：{root / exp / str(seed) / 'status.pth'}"
        )

    model_store.load_vocab()
    model = build_model(env, training_status, config)

    props = set(env.get_propositions())
    print("环境原子命题：", props)
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=render)

    num_successes = 0
    num_violations = 0
    num_accepting_visits = 0
    steps: List[int] = []
    rets: List[float] = []

    # 设一次种子即可
    try:
        env.reset(seed=seed)
    except TypeError:
        pass

    iterator = range(num_episodes)
    if not render:
        iterator = tqdm(iterator)

    for i in iterator:
        obs, info = _env_reset(env)
        if render and isinstance(obs, dict) and 'goal' in obs:
            print(obs['goal'])
        agent.reset()

        done = False
        num_steps = 0
        while not done:
            action = agent.get_action(obs, info, deterministic=deterministic)
            action = np.array(action).flatten()
            if action.shape == (1,):
                action = action[0]

            obs, reward, done, info = _env_step(env, action)
            num_steps += 1

            if done:
                if finite:
                    final_reward = int('success' in info)
                    if 'success' in info:
                        num_successes += 1
                        steps.append(num_steps)
                    elif 'violation' in info:
                        num_violations += 1
                    rets.append(final_reward * (gamma ** max(0, num_steps - 1)))
                    if not render:
                        iterator.set_postfix({
                            'S': num_successes / (i + 1),     # 成功率
                            'V': num_violations / (i + 1),    # 违规率
                            'ADR': float(np.mean(rets)) if rets else 0.0,
                            'AS': float(np.mean(steps)) if steps else 0.0,
                        })
                else:
                    # 无限时域：记录接受访问次数
                    num_accepting_visits += int(info.get('num_accepting_visits', 0))
                    if not render:
                        iterator.set_postfix({
                            'A': num_accepting_visits / (i + 1),
                        })

    env.close()
    if finite:
        success_rate = num_successes / num_episodes
        violation_rate = num_violations / num_episodes
        average_steps = float(np.mean(steps)) if steps else float('nan')
        adr = float(np.mean(rets)) if rets else float('nan')
        print(f'{seed}: SR={success_rate:.3f}, VR={violation_rate:.3f}, ADR={adr:.3f}, AS={average_steps:.3f}')
        return success_rate, average_steps
    else:
        average_visits = num_accepting_visits / num_episodes
        print(f'{seed}: A={average_visits:.3f}')
        return average_visits


def _run_manual():
    """不带命令行时，使用手动配置区运行。"""
    env_name = ENV_NAME
    gamma = _gamma_for_env(env_name)

    exps = _ensure_exps(env_name, EXPS, SEED, AUTO_PICK, AUTO_PICK_K)

    results = []
    for exp in exps:
        metric = simulate(env_name, gamma, exp, SEED, NUM_EPISODES,
                          FORMULA, FINITE, RENDER, DETERMINISTIC)
        results.append((exp, metric))

    # 打印对比
    if FINITE:
        print(f'{"EXP":<36}  {"SR(成功率)":>10}  {"AS(平均步数)":>12}')
        for exp, (sr, avg_steps) in results:
            print(f'{exp:<36}  {sr:>10.3f}  {avg_steps:>12.1f}')
    else:
        print(f'{"EXP":<36}  {"A(平均接受访问次数)":>22}')
        for exp, avg_visits in results:
            print(f'{exp:<36}  {avg_visits:>22.3f}')

    return results


def _run_with_argparse():
    """兼容原来的命令行用法（有参数时才走这里）。"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str,
                        choices=['PointLtl2-v0', 'LetterEnv-v0', 'FlatWorld-v0'],
                        default=ENV_NAME)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--exp', type=str, default=None,
                       help='单实验名；与 --exps 互斥')
    group.add_argument('--exps', type=str, nargs='+',
                       help='多个实验名，空格分隔')

    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--num_episodes', type=int, default=NUM_EPISODES)
    parser.add_argument('--formula', type=str, default=FORMULA)
    parser.add_argument('--finite', action=argparse.BooleanOptionalAction, default=FINITE)
    parser.add_argument('--render', action=argparse.BooleanOptionalAction, default=RENDER)
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=DETERMINISTIC)
    args = parser.parse_args()

    gamma = _gamma_for_env(args.env)
    exps = args.exps if args.exps else ([args.exp] if args.exp else None)
    exps = _ensure_exps(args.env, exps, args.seed, AUTO_PICK, AUTO_PICK_K)

    results = []
    for exp in exps:
        metric = simulate(args.env, gamma, exp, args.seed, args.num_episodes,
                          args.formula, args.finite, args.render, args.deterministic)
        results.append((exp, metric))

    if args.finite:
        print(f'{"EXP":<36}  {"SR(成功率)":>10}  {"AS(平均步数)":>12}')
        for exp, (sr, avg_steps) in results:
            print(f'{exp:<36}  {sr:>10.3f}  {avg_steps:>12.1f}')
    else:
        print(f'{"EXP":<36}  {"A(平均接受访问次数)":>22}')
        for exp, avg_visits in results:
            print(f'{exp:<36}  {avg_visits:>22.3f}')

    return results


def main():
    # 没有命令行参数：走手动配置；有参数：走 argparse
    if len(sys.argv) == 1:
        _run_manual()
    else:
        _run_with_argparse()


if __name__ == '__main__':
    main()
