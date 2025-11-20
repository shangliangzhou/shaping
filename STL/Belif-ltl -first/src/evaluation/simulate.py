import random
import argparse
import numpy as np
import torch
from tqdm import tqdm
import sys

# 按你的项目路径调整（保留你给的原路径设置）
sys.path.append('/home/gh/公共/zh/test/STL/Belif-ltl/src')

from envs import make_env
from ltl import FixedSampler
from model.model import build_model
from model.agent import Agent
from config import model_configs
from sequence.search import ExhaustiveSearch
from utils.model_store import ModelStore


# ------------------------------
# 工具函数：兼容 Gym / Gymnasium
# ------------------------------
def _env_reset(env, seed=None):
    """兼容 Gym / Gymnasium 的 reset() 返回 (obs, info) 或 obs。"""
    try:
        if seed is not None:
            out = env.reset(seed=seed)
        else:
            out = env.reset()
    except TypeError:
        # 某些环境没有 seed 参数
        out = env.reset()

    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
    else:
        obs, info = out, {}
    return obs, info


def _env_step(env, action):
    """兼容 Gym / Gymnasium 的 step() 返回 4 元组或 5 元组。"""
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, reward, done, info
    elif isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, done, info
    else:
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

    # 公式：默认有限时域 Reach-Avoid 示例；也可切换到 Büchi（GF/G）示例
    # parser.add_argument('--formula', type=str,
    #                     default='(F blue) & (!blue U (green & F yellow)) & (! magenta)')
    parser.add_argument('--formula', type=str,
                        default='GF blue & GF green & GF yellow & G !magenta')

    # 默认采用有限轨迹模拟
    parser.add_argument('--finite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--render', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True)
    # >>> 新增：回合上限 <<<
    parser.add_argument('--max_steps', type=int, default=None,
                        help='每回合最大步数；设置后覆盖环境默认上限（例如 300 或 1000）。')

    # ====== 评测端噪声（C→N）开关 ======
    parser.add_argument('--eval_noise_enable', action=argparse.BooleanOptionalAction, default=False,
                        help='仅评测端对 info["propositions"]（集合）注入噪声（漏报/误报）。')
    parser.add_argument('--noise_p_miss', type=float, default=0.0,
                        help='漏报概率：对当前为真的命题，以该概率删掉。')
    parser.add_argument('--noise_p_false', type=float, default=0.0,
                        help='误报概率：对当前为假的命题（全集内），以该概率加入。')
    parser.add_argument('--noise_seed', type=int, default=None,
                        help='噪声随机种子（仅评测端）。')
    parser.add_argument('--noise_log_interval', type=int, default=0,
                        help='>0 则每隔 k 步打印一次噪声统计与 raw→noisy 的命题集合对照。')

    args = parser.parse_args()

    # 折扣因子：与原项目一致
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
        eval_noise_enable=args.eval_noise_enable,
        noise_p_miss=args.noise_p_miss,
        noise_p_false=args.noise_p_false,
        noise_seed=args.noise_seed,
        noise_log_interval=args.noise_log_interval,
        max_steps=args.max_steps
    )


def simulate(env, gamma, exp, seed, num_episodes, formula, finite, render, deterministic,
             eval_noise_enable=False, noise_p_miss=0.0, noise_p_false=0.0,
             noise_seed=None, noise_log_interval=0,max_steps=None):

    # 设随机种子（确保可复现）
    env_name = env
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # 固定一条公式
    sampler = FixedSampler.partial(formula)

    # ====== 构建环境并透传“评测端噪声”参数 ======
    # 若你的 make_env 尚未添加这些 kwargs，会触发 TypeError；我们回退到无噪声构建，保证脚本可用。
    try:
        env = make_env(
            env_name,
            sampler,
            render_mode='human' if render else None,
            eval_noise_enable=eval_noise_enable,
            noise_p_miss=noise_p_miss,
            noise_p_false=noise_p_false,
            noise_seed=noise_seed,
            max_steps=max_steps
        )
    except TypeError:
        # 兼容旧版 make_env（无噪声参数）
        env = make_env(env_name, sampler, render_mode='human' if render else None)
        if eval_noise_enable and (noise_p_miss > 0.0 or noise_p_false > 0.0):
            print("[WARN] make_env 不接受噪声参数。请确认已在 LTLWrapper 与 LDBAWrapper 之间接入 PropositionNoiseWrapper。")

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
        # 兼容：若环境不提供接口，则从 sampler 或配置推断（此处做兜底）
        props = set(getattr(config, "propositions", []))
    print("环境原子命题：", props)

    # 策略与搜索器
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=render)

    # 记录器
    num_successes = 0
    num_violations = 0
    num_accepting_visits = 0
    steps = []
    rets = []

    # 初次 reset（设种子）
    _ = _env_reset(env, seed=seed)

    pbar = range(num_episodes)
    if not render:
        pbar = tqdm(pbar)

    for i in pbar:
        obs, info = _env_reset(env)  # 不再每回合设种子，避免轨迹完全相同
        if render and isinstance(obs, dict) and ('goal' in obs):
            print(obs['goal'])
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

            # 可选：打印噪声统计，确认 C→N 生效（仅当包装器注入了 noise_stats）
            if noise_log_interval > 0 and ('noise_stats' in info) and (num_steps % noise_log_interval == 0):
                ns = info['noise_stats']
                raw = info.get('propositions_raw', None)
                noisy = info.get('propositions_noisy', None)
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

                    # 折扣回报（只在成功时按剩余步折扣）
                    rets.append(final_reward * (gamma ** max(0, num_steps - 1)))

                    if not render:
                        # 训练后评测进度条显示
                        pbar.set_postfix({
                            'S': num_successes / (i + 1),           # 成功率
                            'V': num_violations / (i + 1),          # 违规率
                            'ADR': np.mean(rets) if rets else 0.0,  # 平均折扣回报
                            'AS': np.mean(steps) if steps else 0.0  # 平均步数
                        })
                else:
                    # 无限时域：接受状态访问次数
                    num_accepting_visits += info.get('num_accepting_visits', 0)
                    if not render:
                        pbar.set_postfix({
                            'A': num_accepting_visits / (i + 1)
                        })

    env.close()

    # 输出评测摘要
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
