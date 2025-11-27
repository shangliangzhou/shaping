#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自测（基线 vs. 带信念）在 LDBA 下的评测脚本（兼容旧版 LDBAWrapper）：
- 使用 GoalStringWrapper 注入 LTL 公式到 obs['goal']（spaces.Text）
- belief 分支：L_stab → propositions 的别名，供 LDBA 使用
- SafetyGymWrapper → (Noise) → (Belief+Alias) → GoalStringWrapper → LDBAWrapper → TimeLimit → RemoveTrunc
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import gymnasium
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

# ---------- 路径 ----------
HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[1]  # .../src/smoke_tests -> .../src
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    print("[DEBUG] add SRC_DIR:", SRC_DIR)

# ---------- 本项目模块 ----------
from envs.remove_trunc_wrapper import RemoveTruncWrapper
from envs.zones.safety_gym_wrapper import SafetyGymWrapper
from envs.ldba_wrapper import LDBAWrapper
try:
    from envs.proposition_noise_wrapper import PropositionNoiseWrapper
except Exception:
    PropositionNoiseWrapper = None
from envs.belief_wrapper import BeliefWrapper, BeliefParams


# ---------- 小工具 ----------
def type_name(x: Any) -> str:
    try:
        return type(x).__name__
    except Exception:
        return str(type(x))

def debug_wrapper_chain(env):
    i, cur = 0, env
    while True:
        print(f"[chain {i}] {type_name(cur)}")
        if hasattr(cur, "env"):
            cur = cur.env
            i += 1
        else:
            break

def unpack_reset(ret):
    # 期望 (obs, info)，但也兼容 (obs,) 或 obs
    if isinstance(ret, tuple):
        if len(ret) == 2:
            return ret
        if len(ret) == 1:
            return ret[0], {}
    if ret is not None:
        return ret, {}
    raise RuntimeError("reset 返回不符合期望 (obs, info)。")


def unpack_step(ret):
    if isinstance(ret, tuple) and len(ret) == 5:
        return ret
    raise RuntimeError("step 返回不符合期望 (obs, reward, terminated, truncated, info)。")

def zero_action(action_space) -> np.ndarray:
    return np.zeros_like(action_space.sample(), dtype=float)

def find_method_downstream(env, method_name):
    cur = env
    while True:
        if hasattr(cur, method_name):
            return getattr(cur, method_name)
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            break
    return None

def angle_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------- Label 别名包装（L_stab -> propositions） ----------
class LabelAliasWrapper(gymnasium.Wrapper):
    def __init__(self, env, in_key: str = "L_stab", out_key: str = "propositions"):
        super().__init__(env)
        self.in_key = in_key
        self.out_key = out_key

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._alias(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._alias(info)
        return obs, reward, terminated, truncated, info

    def _alias(self, info: Dict[str, Any]):
        src = info.get(self.in_key, None)
        if src is not None:
            if not isinstance(src, (set, list, tuple)):
                try:
                    src = set(src)
                except Exception:
                    src = set()
            info[self.out_key] = set(src)


# ---------- Goal 字符串注入（obs['goal'] 为 LTL 公式） ----------
class GoalStringWrapper(gymnasium.ObservationWrapper):
    """
    保证观测包含 'goal' 且为字符串（spaces.Text）：供 LDBAWrapper.construct_ldba 使用。
    """
    def __init__(self, env, goal_formula: str, max_len: int = 256):
        super().__init__(env)
        self.goal_formula = str(goal_formula)
        if not isinstance(self.observation_space, spaces.Dict):
            raise TypeError("GoalStringWrapper 期望 Dict 观测空间。")

        # 使用 Gymnasium 的 Text 空间描述字符串
        if hasattr(spaces, "Text"):
            goal_space = spaces.Text(max_length=max_len)
        else:
            # 极少数旧版本兜底：仍然声明为 Text-like；LDBA 只读取值，不强校验
            class _FakeText(spaces.Space):
                def __init__(self): super().__init__((), None, None)
                def sample(self): return ""
                def contains(self, x): return isinstance(x, str)
            goal_space = _FakeText()

        new_spaces = dict(self.observation_space.spaces)
        new_spaces['goal'] = goal_space
        self.observation_space = spaces.Dict(new_spaces)

    def observation(self, observation):
        observation['goal'] = self.goal_formula
        return observation


# ---------- 环境构造 ----------
def make_eval_env(env_id: str,
                  *,
                  goal_formula: str,
                  with_belief: bool,
                  noise_enable: bool,
                  noise_p_miss: float,
                  noise_p_false: float,
                  max_steps: int,
                  prefer_label_key: str = "L_stab",
                  debug_chain: bool = True):
    import safety_gymnasium
    base = safety_gymnasium.make(env_id, render_mode=None)

    # 1) 几何 + sdist
    env = SafetyGymWrapper(base, wall_sensor=True)

    # 2) 可选命题噪声（翻转瞬时 propositions）
    if noise_enable and PropositionNoiseWrapper is not None:
        env = PropositionNoiseWrapper(env,
                                      p_miss=noise_p_miss,
                                      p_false=noise_p_false,
                                      in_key='propositions',
                                      out_key='propositions')

    # 3) 信念层：输出 L_stab；并别名为 propositions 供 LDBA 使用
    if with_belief:
        params = BeliefParams(
            k=5.0, m=0.0,
            ema_lambda=0.8, theta_up=0.8, theta_down=0.2,
            k_stab=3
        )
        env = BeliefWrapper(env, params=params, in_key='propositions_geom', out_key='L_stab')
        env = LabelAliasWrapper(env, in_key=prefer_label_key, out_key='propositions')

    # 4) 注入 LTL 公式到 obs['goal']（字符串）
    env = GoalStringWrapper(env, goal_formula=goal_formula, max_len=256)

    # 5) LDBA（旧签名，不传 sampler）
    env = LDBAWrapper(env)

    # 6) 限步 + 去 trunc
    env = TimeLimit(env, max_episode_steps=max_steps)
    env = RemoveTruncWrapper(env)

    if debug_chain:
        debug_wrapper_chain(env)
    return env


# ---------- 驶入控制器（用于把 agent 开到指定色块附近，方便触发） ----------
def get_agent_xy_readonly(env) -> Optional[np.ndarray]:
    pos = getattr(env.unwrapped, "agent_pos", None)
    if pos is None:
        return None
    pos = np.asarray(pos, dtype=float)
    if pos.size < 2:
        return None
    return pos[:2]

def get_agent_heading_readonly(env) -> float:
    ang = getattr(env.unwrapped, "agent_rot", None)
    try:
        return float(ang)
    except Exception:
        return 0.0

def resolve_color_center(env, color: str) -> Optional[Tuple[float, float]]:
    get_zones_truth = find_method_downstream(env, "get_zones_truth")
    if get_zones_truth is None:
        return None
    try:
        zones = get_zones_truth()
        lst = zones.get(color, [])
        if lst:
            c = lst[0].center_xy
            return float(c[0]), float(c[1])
    except Exception:
        pass
    return None

def drive_to_color(env, color: str,
                   speed: float = 0.7,
                   turn_gain: float = 2.0,
                   max_steps: int = 300) -> bool:
    center = resolve_color_center(env, color)
    if center is not None:
        print(f"[DRIVE] target={color} center=({center[0]:.3f},{center[1]:.3f})")
        cx, cy = center
        stab_need = 3
        stab_count = 0
        for t in range(max_steps):
            xy = get_agent_xy_readonly(env)
            if xy is None:
                obs, reward, terminated, truncated, info = unpack_step(env.step(zero_action(env.action_space)))
                s = float(info.get(f"sdist_{color}", 0.0))
                print(f"  drive {t:03d}: dist=nan sdist_{color}={s:.3f} geom={set(info.get('propositions_geom',set()))}")
                break

            vec = np.array([cx, cy]) - np.array(xy)
            dist = float(np.linalg.norm(vec))
            heading = get_agent_heading_readonly(env)
            target_ang = float(np.arctan2(vec[1], vec[0]))
            err = angle_wrap(target_ang - heading)
            a = np.array([speed, turn_gain * err], dtype=float)
            a = np.clip(a, -1.0, 1.0)
            obs, reward, terminated, truncated, info = unpack_step(env.step(a))
            s = float(info.get(f"sdist_{color}", 0.0))
            geom = set(info.get('propositions_geom', set()))
            noisy = set(info.get('propositions', set()))
            print(f"  drive {t:03d}: dist={dist:.3f} sdist_{color}={s:.4f} geom={geom} noisy={noisy}")
            if s > 0.0:
                stab_count += 1
                if stab_count >= stab_need:
                    return True
            else:
                stab_count = 0
        return False

    print(f"[DRIVE-SDIST] target={color}")
    for t in range(max_steps):
        obs, reward, terminated, truncated, info = unpack_step(env.step(zero_action(env.action_space)))
        s = float(info.get(f"sdist_{color}", 0.0))
        geom = set(info.get('propositions_geom', set()))
        noisy = set(info.get('propositions', set()))
        print(f"  drive {t:03d}: sdist_{color}={s:.4f} geom={geom} noisy={noisy}")
        if s > 0.0:
            return True
    return False


# ---------- 单回合评测 ----------
def run_episode(env, max_steps: int, *, skip_reset: bool = False) -> Tuple[Dict[str, float], str]:
    steps = 0
    success = 0
    violation = 0
    accepting_visits = 0
    diff_cnt = 0
    used_label = None

    if not skip_reset:
        obs, info = unpack_reset(env.reset(seed=None))
    else:
        obs, info = None, {}

    for t in range(max_steps):
        if used_label is None:
            used_label = 'L_stab' if 'L_stab' in info else 'propositions'

        geom = set(info.get('propositions_geom', set()))
        lab = set(info.get(used_label, set()))
        if geom != lab:
            diff_cnt += 1

        accepting_visits += int(info.get('num_accepting_visits', 0))
        if info.get('ltl_success', False) or info.get('success', False):
            success = 1
        if info.get('ltl_violation', False) or int(info.get('violation', 0)) > 0:
            violation = 1

        action = zero_action(env.action_space)
        obs, reward, terminated, truncated, info = unpack_step(env.step(action))
        steps += 1

        if terminated or truncated:
            if info.get('ltl_success', False) or info.get('success', False):
                success = 1
            if info.get('ltl_violation', False) or int(info.get('violation', 0)) > 0:
                violation = 1
            break

    stats = {
        'steps': steps,
        'success': success,
        'violation': violation,
        'accepting_visits': accepting_visits,
        'diff_steps': diff_cnt
    }
    return stats, (used_label or 'propositions')


# ---------- 批量运行 ----------
def summarize(name: str, records: List[Tuple[Dict[str, float], str]]):
    if not records:
        return
    sr = np.mean([r[0]['success'] for r in records])
    viol = np.mean([r[0]['violation'] for r in records])
    acc = np.mean([r[0]['accepting_visits'] for r in records])
    avg_steps = np.mean([r[0]['steps'] for r in records])
    diff_rate = np.mean([r[0]['diff_steps'] / max(1, r[0]['steps']) for r in records])
    used = records[0][1]
    print(f"[SUMMARY] {name} -> SR={sr:.2f}  Viol={viol:.2f}  AccVisits={acc:.2f}  "
          f"AvgSteps={avg_steps:.1f}  DiffRate={diff_rate:.3f}  Label={used}")

def _default_formula(drive_color: Optional[str]) -> str:
    if drive_color:
        return f"F {drive_color}"
    return "F yellow"

def run_suite(env_id: str,
              episodes: int,
              steps: int,
              drive_color: Optional[str],
              *,
              goal_formula: Optional[str],
              noise_enable: bool,
              p_miss: float,
              p_false: float,
              with_belief: bool,
              max_drive_steps: int = 300):

    formula = goal_formula or _default_formula(drive_color)
    print(f"\n[GOAL] LTL formula: {formula}")

    # baseline
    print(f"\n[RUN] baseline | belief={False} | noise={noise_enable} (miss={p_miss}, false={p_false}) | episodes={episodes}")
    env0 = make_eval_env(env_id,
                         goal_formula=formula,
                         with_belief=False,
                         noise_enable=noise_enable,
                         noise_p_miss=p_miss,
                         noise_p_false=p_false,
                         max_steps=steps,
                         prefer_label_key='L_stab',
                         debug_chain=True)
    rec0: List[Tuple[Dict[str, float], str]] = []
    for ep in range(episodes):
        unpack_reset(env0.reset(seed=None))
        if drive_color:
            _ = drive_to_color(env0, drive_color, max_steps=max_drive_steps)
        stats, used_label = run_episode(env0, steps, skip_reset=True)
        print(f"  EP{ep:02d}: {stats} | label_used={used_label}")
        rec0.append((stats, used_label))
    summarize("baseline", rec0)

    # belief
    if with_belief:
        print(f"\n[RUN] belief | belief={True} | noise={noise_enable} (miss={p_miss}, false={p_false}) | episodes={episodes}")
        env1 = make_eval_env(env_id,
                             goal_formula=formula,
                             with_belief=True,
                             noise_enable=noise_enable,
                             noise_p_miss=p_miss,
                             noise_p_false=p_false,
                             max_steps=steps,
                             prefer_label_key='L_stab',
                             debug_chain=True)
        rec1: List[Tuple[Dict[str, float], str]] = []
        for ep in range(episodes):
            unpack_reset(env1.reset(seed=None))
            if drive_color:
                _ = drive_to_color(env1, drive_color, max_steps=max_drive_steps)
            stats, used_label = run_episode(env1, steps, skip_reset=True)
            print(f"  EP{ep:02d}: {stats} | label_used={used_label}")
            rec1.append((stats, used_label))
        summarize("belief", rec1)


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="PointLtl2-v0")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--drive", type=str, default=None, help="可选: blue|green|yellow|magenta")
    parser.add_argument("--formula", type=str, default=None, help="LTL 公式（缺省为 F <drive_color>，若无 drive 则 F yellow）")
    parser.add_argument("--with_belief", action="store_true")
    parser.add_argument("--p_miss", type=float, default=0.0)
    parser.add_argument("--p_false", type=float, default=0.0)
    args = parser.parse_args()

    run_suite(env_id=args.env,
              episodes=args.episodes,
              steps=args.steps,
              drive_color=args.drive,
              goal_formula=args.formula,
              noise_enable=(args.p_miss > 0.0 or args.p_false > 0.0),
              p_miss=args.p_miss,
              p_false=args.p_false,
              with_belief=args.with_belief)

if __name__ == "__main__":
    main()
