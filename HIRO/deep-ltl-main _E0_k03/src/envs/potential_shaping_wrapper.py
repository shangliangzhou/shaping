# src/envs/potential_shaping_wrapper.py
from __future__ import annotations

try:
    import gymnasium as gym
except Exception:
    import gym

from typing import Any, Dict, Iterable, Set, Tuple, Union
import numpy as np


def _as_tuple_reset(out):
    if isinstance(out, tuple):
        if len(out) == 2:
            return out[0], out[1]
        return out[0], {}
    else:
        return out, {}


def _as_tuple_step(out):
    if isinstance(out, tuple):
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
            return obs, float(reward), done, info
        elif len(out) == 4:
            obs, reward, done, info = out
            return obs, float(reward), bool(done), info
        else:
            obs = out[0]
            reward = float(out[1]) if len(out) > 1 else 0.0
            done = bool(out[2]) if len(out) > 2 else False
            info = out[3] if len(out) > 3 else {}
            return obs, reward, done, info
    else:
        return out, 0.0, False, {}


class PotentialShapingWrapper(gym.Wrapper):
    r"""
    训练期的势函数奖励塑形（等价势） + 可选“路点密化”项（κ）。

    记：
      remain(s) = len(obs['goal'])
      d_goal(s) = 1 - |P ∩ G0| / |G0|
         - P = set(obs['propositions'])
         - G0 = 当前下一个目标命题集合（若为单命题则 |G0|=1）

      Φ(s) = - ( remain(s) + α * d_goal(s) )

    形塑：
      r' = r + β ( γ Φ(s') - Φ(s) ) + η · 1{remain(s') < remain(s)} + κ (d_wp(s) - d_wp(s'))
      其中 d_wp 为到路点的归一化距离（由 WaypointWrapper 在 info 中提供 prev/next）。

    旧接口兼容：
      - reset() 仅返回 obs
      - step()  返回 (obs, reward, done, info)
    """
    def __init__(self,
                 env: gym.Env,
                 beta: float = 0.01,
                 eta: float = 0.02,
                 alpha: float = 0.0,
                 kappa: float = 0.0,
                 gamma: float = 0.998):
        super().__init__(env)
        self.beta = float(beta)
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.kappa = float(kappa)
        self.gamma = float(gamma)

        # 缓存上一步势值与 remain
        self._last_phi = None
        self._last_remain = None

        # 统计日志
        self._shape_sum = 0.0
        self._prog_hits = 0

        # 首次 reset 打印一次
        self._printed = False

    # ---------- 日志 ----------
    def get_shaping_stats_and_reset(self) -> Dict[str, float]:
        out = dict(step_shaping_sum=float(self._shape_sum),
                   progress_hits=int(self._prog_hits))
        self._shape_sum = 0.0
        self._prog_hits = 0
        return out

    # ---------- 重载 ----------
    def reset(self, **kwargs):
        raw = self.env.reset(**kwargs)
        obs, info = _as_tuple_reset(raw)
        # 初始化 last_phi/last_remain
        phi, remain = self._phi_and_remain(obs)
        self._last_phi = float(phi)
        self._last_remain = int(remain)
        if not self._printed:
            self._printed = True
            dgoal0 = self._d_goal(obs)
            print(f"[PotentialShapingWrapper] activated: "
                  f"beta={self.beta}, eta={self.eta}, gamma={self.gamma}, "
                  f"alpha={self.alpha}, remain0={self._last_remain}, dgoal0={dgoal0}")
        # 旧接口：仅返回 obs，避免 tuple 进入上层
        return obs

    def step(self, action):
        raw = self.env.step(action)
        obs, reward, done, info = _as_tuple_step(raw)

        # 势差（等价）
        phi_next, remain_next = self._phi_and_remain(obs)
        shaping = 0.0
        if self._last_phi is not None:
            shaping += self.beta * (self.gamma * float(phi_next) - float(self._last_phi))

        # 进度脉冲
        if self._last_remain is not None and remain_next < self._last_remain:
            shaping += self.eta
            self._prog_hits += 1

        # 路点密化（非等价，温和密化）
        if "dist_wp_prev" in info and "dist_wp_next" in info and self.kappa != 0.0:
            shaping += self.kappa * (float(info["dist_wp_prev"]) - float(info["dist_wp_next"]))

        reward = float(reward) + float(shaping)
        self._shape_sum += float(shaping)

        # 更新缓存
        self._last_phi = float(phi_next)
        self._last_remain = int(remain_next)

        # 旧接口：返回 4 元组
        return obs, reward, done, info

    # ---------- 势函数相关 ----------
    def _phi_and_remain(self, obs: Dict[str, Any]) -> Tuple[float, int]:
        remain = self._remain(obs)
        dgoal = self._d_goal(obs)
        phi = - (float(remain) + self.alpha * float(dgoal))
        return float(phi), int(remain)

    @staticmethod
    def _remain(obs: Dict[str, Any]) -> int:
        goal = obs.get("goal", [])
        try:
            return int(len(goal))
        except Exception:
            return 0

    @staticmethod
    def _as_set(x) -> Set[Any]:
        if x is None:
            return set()
        if isinstance(x, (list, tuple, set)):
            return set(x)
        return {x}

    def _d_goal(self, obs: Dict[str, Any]) -> float:
        """1 - |P ∩ G0| / |G0|，G0 为下一个目标命题集合（长度为 1 的情况等价于 bool 匹配）。"""
        goal_seq = obs.get("goal", [])
        if goal_seq is None or len(goal_seq) == 0:
            return 0.0
        g0 = goal_seq[0]
        G0 = self._as_set(g0)
        if len(G0) == 0:
            return 0.0
        P = self._as_set(obs.get("propositions", []))
        inter = len(P.intersection(G0))
        return float(1.0 - inter / len(G0))
