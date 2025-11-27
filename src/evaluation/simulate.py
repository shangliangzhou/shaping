#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import argparse
import numpy as np
import torch
from tqdm import tqdm

from envs import make_env
from ltl import FixedSampler
from model.model import build_model
from model.agent import Agent
from config import model_configs
from sequence.search import ExhaustiveSearch
from utils.model_store import ModelStore


# ------------------------------
# 兼容 Gym / Gymnasium
# ------------------------------
def _env_reset(env, seed=None):
    try:
        out = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, {}


def _env_step(env, action):
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, bool(terminated or truncated), info
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, bool(done), info
    raise RuntimeError("Unexpected env.step() return format.")


# ------------------------------
# 主入口
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str,
                        choices=['PointLtl2-v0', 'LetterEnv-v0', 'FlatWorld-v0'],
                        default='PointLtl2-v0')
    parser.add_argument('--exp', type=str, default='zones_paper_s1')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_episodes', type=int, default=500)

    # 公式：默认无限时域（也可切换到有限时域示例）
    # parser.add_argument('--formula', type=str,
    #                     default='GF blue & GF green & GF yellow & G !magenta') F a ∧ F b ∧ F c
    # parser.add_argument('--formula', type=str, default='(F blue) & (!blue U (green & F yellow)) & (magenta)')
    parser.add_argument('--formula', type=str, default='F (blue & F (green & F yellow))')
    parser.add_argument('--finite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--render', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True)

    # 每回合步数上限（覆盖环境默认值）
    parser.add_argument('--max_steps', type=int, default=None)

    # ====== 评测端噪声（C→N） ======
    parser.add_argument('--eval_noise_enable', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--noise_p_miss', type=float, default=0.0)
    parser.add_argument('--noise_p_false', type=float, default=0.0)
    parser.add_argument('--noise_seed', type=int, default=None)
    parser.add_argument('--noise_log_interval', type=int, default=0)

    # ====== BeliefWrapper（命题信念化 / 稳定门）======
    parser.add_argument('--belief_enable', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--belief_theta_up', type=float, default=0.8)
    parser.add_argument('--belief_theta_down', type=float, default=0.2)
    parser.add_argument('--belief_k_stab', type=int, default=3)
    parser.add_argument('--belief_ema_lambda', type=float, default=0.8)
    parser.add_argument('--belief_k', type=float, default=5.0)
    parser.add_argument('--belief_m', type=float, default=0.0)

    # ====== 基线的 k-stab（若你想给基线也加“耐心门”作公平对比）======
    # parser.add_argument('--baseline_k_stab_enable', action=argparse.BooleanOptionalAction, default=False)
    # parser.add_argument('--baseline_k_stab', type=int, default=3)

    args = parser.parse_args()

    # 折扣（与项目一致）
    if args.env == 'LetterEnv-v0':
        gamma = 0.94
    elif args.env == 'PointLtl2-v0':
        gamma = 0.998
    else:
        gamma = 0.98

    return simulate(
        env=args.env,
        gamma=gamma,
        exp=args.exp,
        seed=args.seed,
        num_episodes=args.num_episodes,
        formula=args.formula,
        finite=args.finite,
        render=args.render,
        deterministic=args.deterministic,
        max_steps=args.max_steps,
        # 噪声
        eval_noise_enable=args.eval_noise_enable,
        noise_p_miss=args.noise_p_miss,
        noise_p_false=args.noise_p_false,
        noise_seed=args.noise_seed,
        noise_log_interval=args.noise_log_interval,
        # 信念化
        belief_enable=args.belief_enable,
        belief_theta_up=args.belief_theta_up,
        belief_theta_down=args.belief_theta_down,
        belief_k_stab=args.belief_k_stab,
        belief_ema_lambda=args.belief_ema_lambda,
        belief_k=args.belief_k,
        belief_m=args.belief_m,
        # 基线耐心门
        # baseline_k_stab_enable=args.baseline_k_stab_enable,
        # baseline_k_stab=args.baseline_k_stab,
    )


def simulate(
    env, gamma, exp, seed, num_episodes, formula, finite, render, deterministic,
    max_steps=None,
    # 噪声
    eval_noise_enable=False, noise_p_miss=0.0, noise_p_false=0.0,
    noise_seed=None, noise_log_interval=0,
    # 信念化
    belief_enable=False, belief_theta_up=0.8, belief_theta_down=0.2,
    belief_k_stab=3, belief_ema_lambda=0.8, belief_k=5.0, belief_m=0.0,
    # 基线耐心门
    baseline_k_stab_enable=False, baseline_k_stab=3,
):
    # 随机种子
    env_name = env
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # 固定公式
    sampler = FixedSampler.partial(formula)

    # ====== 构建环境并透传参数 ======
    try:
        env = make_env(
            env_name, sampler,
            render_mode='human' if render else None,
            max_steps=max_steps,
            # 噪声
            eval_noise_enable=eval_noise_enable,
            noise_p_miss=noise_p_miss,
            noise_p_false=noise_p_false,
            noise_seed=noise_seed,
            # 信念化
            belief_enable=belief_enable,
            belief_theta_up=belief_theta_up,
            belief_theta_down=belief_theta_down,
            belief_k_stab=belief_k_stab,
            belief_ema_lambda=belief_ema_lambda,
            belief_k=belief_k,
            belief_m=belief_m,
            # 基线耐心门（若 make_env 支持）
            # baseline_k_stab_enable=baseline_k_stab_enable,
            # baseline_k_stab=baseline_k_stab,
        )
    except TypeError as e:
        print("[WARN] make_env 不接受某些新参数，将退回最小参数集。详情：", e)
        env = make_env(env_name, sampler, render_mode='human' if render else None)
        if eval_noise_enable and (noise_p_miss > 0.0 or noise_p_false > 0.0):
            print("[WARN] 未能在 make_env 插入 PropositionNoiseWrapper，请确认 env_utils.make_env 已实现。")

    # 模型装载
    config = model_configs[env_name]
    model_store = ModelStore(env_name, exp, seed)
    training_status = model_store.load_training_status(map_location='cpu')
    model_store.load_vocab()
    model = build_model(env, training_status, config)

    # 命题全集（用于 ExhaustiveSearch）
    if hasattr(env, "get_propositions"):
        props = set(env.get_propositions())
    else:
        props = set(getattr(config, "propositions", []))
    print("环境原子命题：", props)

    # 策略与搜索器
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=render)

    # 记录
    num_successes = 0
    num_violations = 0
    num_accepting_visits = 0
    steps = []
    rets = []

    _ = _env_reset(env, seed=seed)

    pbar = range(num_episodes)
    if not render:
        pbar = tqdm(pbar)

    for i in pbar:
        obs, info = _env_reset(env)
        agent.reset()
        done = False
        num_steps = 0

        while not done:
            action = agent.get_action(obs, info, deterministic=deterministic)
            action = action.flatten()
            if action.shape == (1,):
                action = action[0]

            obs, reward, done, info = _env_step(env, action)
            num_steps += 1

            # 噪声日志（若 wrapper 写入了 noise_stats）
            if noise_log_interval > 0 and ('noise_stats' in info) and (num_steps % noise_log_interval == 0):
                ns = info['noise_stats']
                raw = info.get('propositions_raw', None)
                noisy = info.get('propositions', None)
                print(f"[NOISE] step={ns.get('steps')} miss_del={ns.get('miss_del')} "
                      f"false_add={ns.get('false_add')} raw={raw} -> noisy={noisy}")

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
                        pbar.set_postfix({
                            'S': num_successes / (i + 1),
                            'V': num_violations / (i + 1),
                            'ADR': np.mean(rets) if rets else 0.0,
                            'AS': np.mean(steps) if steps else 0.0
                        })
                else:
                    num_accepting_visits += info.get('num_accepting_visits', 0)
                    if not render:
                        pbar.set_postfix({'A': num_accepting_visits / (i + 1)})

    env.close()

    if finite:
        success_rate = num_successes / num_episodes
        violation_rate = num_violations / num_episodes
        average_steps = float(np.mean(steps)) if steps else float('nan')
        adr = float(np.mean(rets)) if rets else 0.0
        print(f'{seed}: {success_rate:.3f},{violation_rate:.3f},{adr:.3f},{average_steps:.3f}')
        return success_rate, average_steps
    else:
        average_visits = num_accepting_visits / num_episodes
        print(f'{seed}: {average_visits:.3f}')
        return average_visits


if __name__ == '__main__':
    main()
