#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速自测：评测期命题噪声注入是否生效 + 与几何真值对比
- 兼容不同 reset/step 返回签名（只取 obs 与 info）
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# ---- 解决导入路径：.../src/smoke_tests -> .../src ----
HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    print("[DEBUG] add SRC_DIR:", SRC_DIR)

from envs import make_env  # 你已重构的无环工厂

# -------- LTL 采样器（固定公式）--------
def fixed_formula_sampler(propositions: list[str]):
    colors = [c for c in ["blue", "green", "yellow", "magenta"] if c in propositions]
    if len(colors) >= 3:
        a, b, c = colors[:3]
        formula = f"(F {a}) & (!{a} U ({b} & F {c}))"
    elif len(colors) == 2:
        a, b = colors
        formula = f"GF {a} & F {b}"
    elif len(colors) == 1:
        a = colors[0]
        formula = f"F {a}"
    else:
        formula = "true"

    def _sample() -> str:
        return formula
    print(f"[DEBUG] fixed_formula_sampler -> {formula}")
    return _sample

# -------- 兼容工具：统一拿到 (obs, info) / (obs, info, ...) 中的前后 --------
def unpack_reset(ret):
    """兼容 reset 返回：obs 或 (obs, info) 或 (obs, ..., info)"""
    if not isinstance(ret, tuple):
        return ret, {}
    if len(ret) == 1:
        return ret[0], {}
    # 多返回时，按规范第一项是 obs，最后一项是 info
    return ret[0], ret[-1]

def unpack_step(ret):
    """兼容 step 返回四元/五元/六元，统一拿到 obs, reward, terminated, truncated, info"""
    if not isinstance(ret, tuple):
        raise RuntimeError("env.step 应返回 tuple")
    # 常见：obs, reward, terminated, truncated, info
    if len(ret) >= 5:
        return ret[0], ret[1], ret[2], ret[3], ret[-1]
    # 旧式：obs, reward, done, info
    if len(ret) == 4:
        obs, reward, done, info = ret
        terminated, truncated = bool(done), False
        return obs, reward, terminated, truncated, info
    raise RuntimeError(f"不支持的 step 返回长度：{len(ret)}")

# -------- 零动作工具（便于稳定观测）--------
def zero_action(space):
    import numpy as np
    from gymnasium import spaces
    if isinstance(space, spaces.Box):
        return np.zeros(space.shape, dtype=space.dtype)
    if isinstance(space, spaces.Discrete):
        return 0
    return space.sample()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="PointLtl2-v0")
    parser.add_argument("--steps", type=int, default=30)
    # 评测期命题噪声
    parser.add_argument("--eval_noise_enable", action="store_true", default=False)
    parser.add_argument("--p_miss", type=float, default=0.20, help="漏检概率")
    parser.add_argument("--p_false", type=float, default=0.05, help="空假率")
    parser.add_argument("--noise_seed", type=int, default=123)
    # 可选：是否启用 BeliefWrapper（若你已接好）
    parser.add_argument("--belief_enable", action="store_true", default=False)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--drive", type=str, default=None,
                    choices=["blue", "green", "yellow", "magenta"],
                    help="若指定，则用简易控制器把 agent 驶入该色区中心附近，再继续打印 diff")
    parser.add_argument("--drive_max_steps", type=int, default=200)
    parser.add_argument("--k_stab_print", type=int, default=5,
                        help="进入色区后额外打印几步，观察稳定性")

    args = parser.parse_args()
    

    env = make_env(
        name=args.env,
        sampler=fixed_formula_sampler,
        sequence=False,
        render_mode="human" if args.render else None,
        # 评测期命题噪声
        eval_noise_enable=args.eval_noise_enable,
        noise_p_miss=args.p_miss,
        noise_p_false=args.p_false,
        noise_seed=args.noise_seed,
        # 可选：信念化
        belief_enable=args.belief_enable,
    )

    # ---- reset（兼容多返回）----
    ret = env.reset(seed=1)
    obs, info = unpack_reset(ret)

    print("\n[RESET]")
    geom = set(info.get("propositions_geom", set()))
    noisy = set(info.get("propositions", set()))
    print("  propositions_geom:", geom)
    print("  propositions     :", noisy)
    print("  sym_diff         :", geom ^ noisy)


    # ---------- 工具：向下游找方法 ----------
    def _find_method_downstream(env, name):
        cur = env
        while hasattr(cur, "env"):
            if hasattr(cur, name):
                return getattr(cur, name)
            cur = cur.env
        return getattr(cur, name, None)

    def _get_agent_xy(env):
        import numpy as np
        pos = getattr(env.unwrapped, "agent_pos", None)
        if pos is None:
            return None
        pos = np.asarray(pos, dtype=float)
        return pos[:2] if pos.size >= 2 else None

    def _get_agent_heading(env):
        import numpy as np
        ang = getattr(env.unwrapped, "agent_rot", None)
        try:
            return float(ang)
        except Exception:
            return 0.0

    def _angle_wrap(a):
        import numpy as np
        return (a + np.pi) % (2*np.pi) - np.pi

    # ---------- 若指定 --drive，则先把 agent 开到目标色区 ----------
    if args.drive is not None:
        get_zones_truth = _find_method_downstream(env, "get_zones_truth")
        if get_zones_truth is None:
            print("[WARN] 环境未暴露 get_zones_truth；跳过驱动。")
        else:
            zones = get_zones_truth()
            if args.drive not in zones or len(zones[args.drive]) == 0:
                print(f"[WARN] 未找到色区几何：{args.drive}；跳过驱动。")
            else:
                # 取该色区第一个几何的中心（SafetyGym 圆区常见）
                z0 = zones[args.drive][0]
                cx, cy = map(float, getattr(z0, "center_xy", (0.0, 0.0)))

                print(f"\n[DRIVE] target={args.drive} center=({cx:.3f},{cy:.3f})")
                speed, turn_gain = 0.7, 2.0
                arrived = False

                for t in range(args.drive_max_steps):
                    import numpy as np
                    xy = _get_agent_xy(env)
                    if xy is None:
                        # 兜底：走一步让底层刷新位姿
                        obs, reward, terminated, truncated, info = unpack_step(env.step(zero_action(env.action_space)))
                        continue

                    vec = np.array([cx, cy], dtype=float) - np.array(xy, dtype=float)
                    dist = float(np.linalg.norm(vec))
                    heading = _get_agent_heading(env)
                    target_ang = float(np.arctan2(vec[1], vec[0]))
                    err = _angle_wrap(target_ang - heading)

                    a = np.array([speed, turn_gain * err], dtype=float)
                    a = np.clip(a, -1.0, 1.0)

                    obs, reward, terminated, truncated, info = unpack_step(env.step(a))
                    geom = set(info.get("propositions_geom", set()))
                    noisy = set(info.get("propositions", set()))
                    s = float(info.get(f"sdist_{args.drive}", 0.0))

                    print(f"  drive {t:03d}: dist={dist:.3f} sdist_{args.drive}={s:.3f} geom={geom} noisy={noisy}")
                    # 进入条件：几何真值含目标色，或该色 sdist>0
                    if (args.drive in geom) or (s > 0):
                        arrived = True
                        break
                    if terminated or truncated:
                        break

                if not arrived:
                    print("[WARN] 未能驶入目标色区；后续仍执行常规步进。")
                else:
                    # 进区后额外走几步，便于观察稳定性/噪声
                    for j in range(args.k_stab_print):
                        obs, reward, terminated, truncated, info = unpack_step(env.step(zero_action(env.action_space)))
                        geom = set(info.get("propositions_geom", set()))
                        noisy = set(info.get("propositions", set()))
                        sd = geom ^ noisy
                        print(f"  post {j:02d}: geom={geom} noisy={noisy} diff={sd}")

                        
    # ---- 连续 step（兼容多返回）----
    for t in range(args.steps):
        a = zero_action(env.action_space)
        obs, reward, terminated, truncated, info = unpack_step(env.step(a))
        geom = set(info.get("propositions_geom", set()))
        noisy = set(info.get("propositions", set()))
        sd = geom ^ noisy

        line = f" step {t:02d}: geom={geom}  noisy={noisy}  diff={sd}"
        # 若开启了信念化且字段存在，额外打印
        if "p_atoms" in info or "b_atoms" in info or "L_stab" in info:
            def _fmt_map(m, k=3):
                try:
                    return {kk: round(float(vv), k) for kk, vv in m.items()}
                except Exception:
                    return m
            p_atoms = _fmt_map(info.get("p_atoms", {}))
            b_atoms = _fmt_map(info.get("b_atoms", {}))
            L_stab = set(info.get("L_stab", set()))
            line += f" | p={p_atoms}  b={b_atoms}  L_stab={L_stab}"
        print(line)

        if terminated or truncated:
            print(" [DONE] terminated or truncated at step", t)
            break

    env.close()

if __name__ == "__main__":
    main()
