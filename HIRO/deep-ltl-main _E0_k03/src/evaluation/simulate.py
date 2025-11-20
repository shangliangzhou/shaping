import random

import numpy as np
import torch
from tqdm import tqdm
import sys
sys.path.append('/home/gh/公共/zh/deep-ltl-main/src')

from envs import make_env
from ltl import FixedSampler
from model.model import build_model
from model.agent import Agent
from config import model_configs
from sequence.search import ExhaustiveSearch
from utils.model_store import ModelStore
import argparse

# 单条 LTL 公式的最小评测器

# 做什么：给定一条公式（有限或无限时域），载入已训练模型，跑若干回合，输出：

# 有限时域：成功率 SR、平均步数 μ、（可选）折扣回报；

# 无限时域：接受状态访问次数/频率（Büchi 接受）、窗口内成功统计等。

# 什么时候用：调公式/调环境/看一次性效果；也是其它批量评测脚本的“原子能力”。

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, choices=['PointLtl2-v0', 'LetterEnv-v0', 'FlatWorld-v0'], default='PointLtl2-v0')
    parser.add_argument('--exp', type=str, default='zones_paper_s1')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--formula', type=str, default='(F blue) & (!blue U (green & F yellow))')
    # parser.add_argument('--formula', type=str, default='GF blue & GF green & GF yellow & G !magenta')
    #默认采用有限轨迹模拟
    parser.add_argument('--finite', action=argparse.BooleanOptionalAction, default=True)
    # parser.add_argument('--finite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--render', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    gamma = 0.94 if args.env == 'LetterEnv-v0' else 0.998 if args.env == 'PointLtl2-v0' else 0.98
    return simulate(args.env, gamma, args.exp, args.seed, args.num_episodes, args.formula, args.finite, args.render, args.deterministic)


def simulate(env, gamma, exp, seed, num_episodes, formula, finite, render, deterministic):
    env_name = env
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    sampler = FixedSampler.partial(formula)
    env = make_env(env_name, sampler, render_mode='human' if render else None)
    config = model_configs[env_name]
    model_store = ModelStore(env_name, exp, seed)
    training_status = model_store.load_training_status(map_location='cpu')
    model_store.load_vocab()
    model = build_model(env, training_status, config)

    props = set(env.get_propositions())
    print("环境原子命题：",props)
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=render)

    num_successes = 0
    num_violations = 0
    num_accepting_visits = 0
    steps = []
    rets = []

    env.reset(seed=seed)

    pbar = range(num_episodes)
    if not render:
        pbar = tqdm(pbar)
    for i in pbar:
        obs, info = env.reset(), {}
        if render:
            print(obs['goal'])
        agent.reset()
        done = False
        num_steps = 0
        while not done:
            action = agent.get_action(obs, info, deterministic=deterministic)
            action = action.flatten()
            if action.shape == (1,):
                action = action[0]
            obs, reward, done, info = env.step(action)
            num_steps += 1
            if done:
                if finite:
                    final_reward = int('success' in info)
                    if 'success' in info:
                        num_successes += 1
                        steps.append(num_steps)
                    elif 'violation' in info:
                        num_violations += 1
                    rets.append(final_reward * gamma ** (num_steps - 1))
                    if not render:
                        pbar.set_postfix({
                            'S': num_successes / (i + 1),  #成功率
                            'V': num_violations / (i + 1),  #违规率
                            'ADR': np.mean(rets), #平均奖励
                            'AS': np.mean(steps), #平均步数
                        })
                else:
                    num_accepting_visits += info['num_accepting_visits']
                    if not render:
                        pbar.set_postfix({
                            'A': num_accepting_visits / (i + 1),
                        })

    env.close()
    if finite:
        success_rate = num_successes / num_episodes
        violation_rate = num_violations / num_episodes
        average_steps = np.mean(steps)
        adr = np.mean(rets)
        print(f'{seed}: {success_rate:.3f},{violation_rate:.3f},{adr:.3f},{average_steps:.3f}')
        return success_rate, average_steps
    else:
        average_visits = num_accepting_visits / num_episodes
        print(f'{seed}: {average_visits:.3f}')
        return average_visits


if __name__ == '__main__':
    main()
