#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速自测 BeliefWrapper 是否工作：
- 打印 reset/step 后的 p_atoms, b_atoms, L_stab
- 自动“驶入”某个色区中心，连续命中稳健门 k_stab 步后判成功
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# --------- 解决导入路径 ----------
HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[1]  # .../src/smoke_tests -> .../src
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    print("[DEBUG] add SRC_DIR:", SRC_DIR)

# --------- 导入项目内包装器 ----------
from envs.zones.safety_gym_wrapper import SafetyGymWrapper
from envs.belief_wrapper import BeliefWrapper, BeliefParams


# --------- 工具函数：向下查找包装器属性/方法 ----------
def find_attr_downstream(env, attr):
    cur = env
    while True:
        if hasattr(cur, attr):
            return getattr(cur, attr)
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            break
    return None


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
    # wrap 到 [-pi, pi]
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="PointLtl2-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5, help="reset 后先走几步做观测")
    parser.add_argument("--k_stab", type=int, default=3, help="稳健门连帧阈值（需与 BeliefParams 一致）")
    parser.add_argument("--theta_up", type=float, default=0.8)
    parser.add_argument("--theta_down", type=float, default=0.2)
    parser.add_argument("--ema_lambda", type=float, default=0.8)
    parser.add_argument("--k", type=float, default=5.0, help="sdist→prob 的温度斜率")
    parser.add_argument("--m", type=float, default=0.0, help="安全边距")
    # 驶入控制器参数
    parser.add_argument("--speed", type=float, default=0.7)
    parser.add_argument("--turn_gain", type=float, default=2.0)
    parser.add_argument("--max_drive_steps", type=int, default=200)
    args = parser.parse_args()

    import safety_gymnasium

    # 1) 构建环境并包两层 wrapper
    base = safety_gymnasium.make(args.env_id, render_mode=None)
    env = SafetyGymWrapper(base, wall_sensor=True)
    params = BeliefParams(
        k=args.k, m=args.m,
        ema_lambda=args.ema_lambda,
        theta_up=args.theta_up,
        theta_down=args.theta_down,
        k_stab=args.k_stab
    )
    env = BeliefWrapper(env, params=params)

    # 2) reset 并打印一次
    obs, info = env.reset(seed=args.seed)
    colors = getattr(env, "colors", ["blue", "green", "yellow", "magenta"])
    print("\n[RESET]")
    print(" agent_xy:", info.get("agent_xy"))
    print(" sdist:", {f"sdist_{c}": round(float(info.get(f"sdist_{c}", 0.0)), 4) for c in colors})
    print(" p_atoms:", {c: round(info["p_atoms"][c], 3) for c in colors})
    print(" b_atoms:", {c: round(info["b_atoms"][c], 3) for c in colors})
    print(" L_stab :", set(info.get("L_stab", set())))
    assert "p_atoms" in info and "b_atoms" in info and "L_stab" in info, "BeliefWrapper outputs missing!"

    # 3) 先走几步，观察数值在变
    print("\n[WALK]")
    for t in range(args.steps):
        a = env.action_space.sample() * 0  # 取零动作便于稳定观测
        obs, reward, terminated, truncated, info = env.step(a)
        print(f" step {t}:",
              "sdist:", {f"sdist_{c}": round(float(info.get(f"sdist_{c}", 0.0)), 3) for c in colors},
              "L_stab:", set(info.get("L_stab", set())))

    # 4) 查找底层几何真值：优先 yellow
    get_zones_truth = find_method_downstream(env, "get_zones_truth")
    target_color, cx, cy = None, None, None
    if get_zones_truth is not None:
        zones = get_zones_truth()
        for cand in ["yellow", "blue", "green", "magenta"]:
            if cand in zones and len(zones[cand]) > 0:
                z = zones[cand][0]
                center = getattr(z, "center_xy", None)
                if center is not None:
                    target_color = cand
                    cx, cy = float(center[0]), float(center[1])
                    break

    print("\n[DRIVE-TO CHECK]")
    if target_color is None:
        print(" !! 未找到几何真值（get_zones_truth 不可用）——跳过驶入测试，仅做行走观测。")
        return
    print(" target:", target_color, "center:", (round(cx, 3), round(cy, 3)))

    # 简单 P 控制器：前进速度常数 + 航向误差成比例转向
    speed = args.speed
    turn_gain = args.turn_gain
    max_steps_drive = args.max_drive_steps
    stab_need = args.k_stab
    stab_count = 0
    entered = False

    def get_agent_xy_readonly(e):
        # 读取只读属性 agent_pos（大多数 builder 暴露）
        pos = getattr(e.unwrapped, "agent_pos", None)
        if pos is None:
            return None
        pos = np.asarray(pos, dtype=float)
        if pos.size < 2:
            return None
        return pos[:2]

    def get_agent_heading_readonly(e):
        # 读取只读属性 agent_rot（若不可用，兜底为 0）
        ang = getattr(e.unwrapped, "agent_rot", None)
        try:
            return float(ang)
        except Exception:
            return 0.0

    for t in range(max_steps_drive):
        xy = get_agent_xy_readonly(env)
        if xy is None:
            # 拿不到位置：退化为打印一次观测后退出
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample() * 0)
            print("  [warn] cannot read agent_xy; sdist snapshot:",
                  {f"sdist_{c}": round(float(info.get(f"sdist_{c}", 0.0)), 3) for c in colors})
            break

        vec = np.array([cx, cy], dtype=float) - np.array(xy, dtype=float)
        dist = float(np.linalg.norm(vec))
        heading = get_agent_heading_readonly(env)
        target_ang = float(np.arctan2(vec[1], vec[0]))
        err = angle_wrap(target_ang - heading)

        # 组装动作（Point 机器人：前进 + 转向），裁剪到 [-1, 1]
        a = np.array([speed, turn_gain * err], dtype=float)
        a = np.clip(a, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(a)

        s = round(float(info.get(f"sdist_{target_color}", 0.0)), 4)
        L = set(info.get("L_stab", set()))
        p = round(info["p_atoms"][target_color], 3)
        b = round(info["b_atoms"][target_color], 3)
        print(f"  drive {t:03d}: dist={dist:.3f} sdist={s} p={p} b={b} L={L}")

        # 稳健门触发统计：在目标区（sdist>0）且 L_stab 命中该色
        if (s > 0.0) and (target_color in L):
            stab_count += 1
            if stab_count >= stab_need:
                entered = True
                break
        else:
            stab_count = 0

    assert entered, (
        f"未能在 {max_steps_drive} 步内稳定进入 {target_color}（需连续 {stab_need} 步 L_stab 命中）。"
        " 可降低 theta_up、减小 k_stab、或增大 speed/turn_gain 试试。"
    )
    print("\n[SUCCESS] BeliefWrapper 工作正常：已通过动作驶入目标色区并触发稳健命题。")


if __name__ == "__main__":
    main()
