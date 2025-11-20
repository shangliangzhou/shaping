# src/envs/waypoint_wrapper.py
from __future__ import annotations

try:
    import gymnasium as gym
except Exception:
    import gym

import numpy as np
from typing import Any, Dict, Tuple, Union


def _as_tuple_reset(out):
    """Normalize Gym/Gymnasium reset outputs to (obs, info)."""
    if isinstance(out, tuple):
        if len(out) == 2:
            return out[0], out[1]
        # Fallback: treat first as obs
        return out[0], {}
    else:
        return out, {}


def _as_tuple_step(out):
    """Normalize Gym/Gymnasium step outputs to (obs, reward, done, info)."""
    if isinstance(out, tuple):
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
            return obs, float(reward), done, info
        elif len(out) == 4:
            obs, reward, done, info = out
            return obs, float(reward), bool(done), info
        else:
            # Unexpected arity; be defensive
            obs = out[0]
            reward = float(out[1]) if len(out) > 1 else 0.0
            done = bool(out[2]) if len(out) > 2 else False
            info = out[3] if len(out) > 3 else {}
            return obs, reward, done, info
    else:
        # Extremely old API; assume obs only
        return out, 0.0, False, {}


class WaypointWrapper(gym.Wrapper):
    """
    给 obs 添加路点 waypoint，并在 info 中输出到路点距离用于密化：
      - obs['waypoint'] = np.array([x, y], float32)
      - info['dist_wp_prev'], info['dist_wp_next'] 归一化距离（/world_scale）

    旧接口兼容：
      - reset() 仅返回 obs
      - step()  返回 (obs, reward, done, info)
    """
    def __init__(self, env: gym.Env, world_scale: float = 1.0, mode: str = "centroid"):
        super().__init__(env)
        self.world_scale = float(world_scale) if world_scale > 0 else 1.0
        self.mode = mode
        self._last_wp = np.zeros(2, dtype=np.float32)
        self._last_wp_dist = None

    # ---------- Gym reset/step 适配 ----------
    def reset(self, **kwargs):
        raw = self.env.reset(**kwargs)
        obs, info = _as_tuple_reset(raw)
        obs = self._attach_waypoint(obs, info)
        self._last_wp_dist = self._wp_dist(obs)
        # reset 按旧接口仅返回 obs，避免 tuple 往上冒
        return obs

    def step(self, action):
        raw = self.env.step(action)
        obs, reward, done, info = _as_tuple_step(raw)

        obs = self._attach_waypoint(obs, info)
        dist_next = self._wp_dist(obs)

        info = dict(info)
        if self._last_wp_dist is None:
            self._last_wp_dist = dist_next
        info["dist_wp_prev"] = float(self._last_wp_dist)
        info["dist_wp_next"] = float(dist_next)
        self._last_wp_dist = dist_next

        # 旧接口：返回 4 元组
        return obs, float(reward), bool(done), info

    # ---------- 内部：计算路点 ----------
    def _attach_waypoint(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        obs = dict(obs)
        wp = self._compute_wp_from_goal(obs.get("goal", []), info)
        self._last_wp = wp.astype(np.float32)
        obs["waypoint"] = self._last_wp
        return obs

    def _compute_wp_from_goal(self, goal_seq, info) -> np.ndarray:
        """
        从“下一个目标命题”推导一个 waypoint。
        - 优先：env.unwrapped.get_zone_polygon(prop) -> (N,2) 顶点，取重心
        - 回退：info['goal_anchor'] (x,y)
        - 兜底：(0,0)
        """
        if goal_seq is None or len(goal_seq) == 0:
            return np.zeros(2, dtype=np.float32)

        g0 = goal_seq[0]  # 允许是 "blue" 或 ["blue","key"] 等
        # 如果是组合命题，先取第一个有定义的区域
        props = g0 if isinstance(g0, (list, tuple, set)) else [g0]

        for p in props:
            try:
                poly = self.env.unwrapped.get_zone_polygon(p)
                poly = np.asarray(poly, dtype=np.float32)
                if poly.ndim == 2 and poly.shape[1] == 2 and poly.shape[0] >= 3:
                    c = poly.mean(axis=0)
                    return c
            except Exception:
                pass

        # 回退：从 info 提供锚点
        if isinstance(info, dict) and "goal_anchor" in info:
            ga = np.asarray(info["goal_anchor"], dtype=np.float32)
            if ga.size >= 2:
                return ga[:2]

        return np.zeros(2, dtype=np.float32)

    def _extract_agent_xy(self, features: Union[np.ndarray, list, tuple]) -> np.ndarray:
        """
        依你的 features 结构取 agent (x,y)。常见：features[:2] 为坐标。
        如有差异，这里按你的环境做定制。
        """
        arr = np.asarray(features, dtype=np.float32)
        if arr.size < 2:
            return np.zeros(2, dtype=np.float32)
        return arr[:2]

    def _wp_dist(self, obs: Dict[str, Any]) -> float:
        agent_xy = self._extract_agent_xy(obs["features"])
        d = np.linalg.norm(agent_xy - self._last_wp) / self.world_scale
        return float(d)
