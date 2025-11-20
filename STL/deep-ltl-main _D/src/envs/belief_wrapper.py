# src/envs/belief_wrapper.py
from __future__ import annotations

try:
    import gymnasium as gym
except Exception:
    import gym

from typing import Any, Dict, Tuple, Union, Optional, Iterable
import math
from perception.detector_model import PropositionDetector

ObsType = Union[Dict[str, Any], Any]


def _H_bernoulli(p: float) -> float:
    eps = 1e-6
    p = min(max(p, eps), 1.0 - eps)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


class BeliefReachAvoidWrapper(gym.Wrapper):
    """
    命题概率化 + 信念推进 + （可选）期望事件脉冲密化。
    对外 API 兼容旧 Gym：step() 始终返回 4 元组 (obs, reward, done, info)。
    reset() 保持与底层一致：若底层是 (obs, info) 则返回 (obs, info)，否则返回 obs。
    """

    def __init__(self,
                 env: gym.Env,
                 ema_tau: float = 0.9,
                 temperature: float = 1.0,
                 reward_coef_delta: float = 0.0,
                 expose_accept_prob_in_obs: bool = False,
                 get_goal_sets=None  # Optional[Callable[[ObsType, Dict], Tuple[Iterable[str], Iterable[str]]]]
                 ):
        super().__init__(env)
        self.detector = PropositionDetector(ema_tau=ema_tau, temperature=temperature)
        self.reward_coef_delta = float(reward_coef_delta)
        self.expose_accept_prob_in_obs = bool(expose_accept_prob_in_obs)
        self.get_goal_sets = get_goal_sets
        self._prev_accept: Optional[float] = None

    # ---------- 内部工具 ----------
    @staticmethod
    def _infer_sets_from_info(info: Dict[str, Any]) -> Tuple[Iterable[str], Iterable[str]]:
        if isinstance(info, dict):
            if "reach_avoid" in info and isinstance(info["reach_avoid"], dict):
                ra = info["reach_avoid"]
                return ra.get("A_plus", []), ra.get("A_minus", [])
            if "ltl" in info and isinstance(info["ltl"], dict):
                ltl = info["ltl"]
                return ltl.get("A_plus", []), ltl.get("A_minus", [])
        return [], []

    @staticmethod
    def _accept_prob(probs: Dict[str, float], A_plus: Iterable[str], A_minus: Iterable[str]) -> float:
        if A_plus:
            p_plus = min(probs.get(p, 0.0) for p in A_plus)  # 合取下界
        else:
            p_plus = 1.0
        if A_minus:
            p_minus = max(probs.get(n, 0.0) for n in A_minus)  # 析取上界
        else:
            p_minus = 0.0
        ap = max(0.0, min(1.0, p_plus * (1.0 - p_minus)))
        return ap

    @staticmethod
    def _belief_entropy(probs: Dict[str, float], atoms: Iterable[str]) -> float:
        atoms = list(atoms)
        if not atoms:
            return 0.0
        return sum(_H_bernoulli(probs.get(a, 0.5)) for a in atoms) / len(atoms)

    # ---------- reset ----------
    def reset(self, **kwargs):
        """
        保持与底层 reset 一致的返回格式：
          - 底层若返回 (obs, info)，则增强 info 后仍返回 (obs, info)
          - 底层若返回 obs，则原样返回 obs（我们内部仍会更新状态）
        """
        out = self.env.reset(**kwargs)

        # 从返回中取 obs 与（可选）info
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}

        # 初始化探测与信念
        self.detector.reset()
        probs = self.detector.predict_proba(obs)
        A_plus, A_minus = (self.get_goal_sets(obs, info) if self.get_goal_sets
                           else self._infer_sets_from_info(info))
        ap = self._accept_prob(probs, A_plus, A_minus)
        self._prev_accept = ap
        atoms = list(set(list(A_plus) + list(A_minus)))

        # 尽量不改变签名：只有在本来就有 info 的情况下我们才写回诊断字段
        if isinstance(out, tuple) and len(out) == 2 and isinstance(info, dict):
            info = dict(info)
            info["accept_prob"] = ap
            info["accept_prob_delta"] = 0.0
            info["belief_entropy"] = self._belief_entropy(probs, atoms)
            if self.expose_accept_prob_in_obs and isinstance(obs, dict):
                obs = dict(obs)
                obs["accept_prob"] = ap
            return obs, info

        # 底层只返回 obs 的情况：不改返回格式
        if self.expose_accept_prob_in_obs and isinstance(obs, dict):
            obs = dict(obs)
            obs["accept_prob"] = ap
        return obs

    # ---------- step ----------
    def step(self, action):
        """
        统一对外返回 4 元组 (obs, reward, done, info)。
        若底层是 Gymnasium 5 元组，折叠 done = terminated or truncated。
        """
        out = self.env.step(action)

        if isinstance(out, tuple) and len(out) == 5:
            obs, r, terminated, truncated, info = out
            done = bool(terminated or truncated)
        elif isinstance(out, tuple) and len(out) == 4:
            obs, r, done, info = out
        else:
            raise RuntimeError("Unexpected env.step() return format")

        # 计算命题概率与接受概率
        probs = self.detector.predict_proba(obs)
        A_plus, A_minus = (self.get_goal_sets(obs, info) if self.get_goal_sets
                           else self._infer_sets_from_info(info))
        ap = self._accept_prob(probs, A_plus, A_minus)
        delta = 0.0 if self._prev_accept is None else (ap - self._prev_accept)
        self._prev_accept = ap

        # 期望事件脉冲密化（可关）
        r_bel = self.reward_coef_delta * float(delta)
        r = float(r) + r_bel

        # 写回诊断字段
        atoms = list(set(list(A_plus) + list(A_minus)))
        info = dict(info) if isinstance(info, dict) else {}
        info["accept_prob"] = ap
        info["accept_prob_delta"] = delta
        info["belief_entropy"] = self._belief_entropy(probs, atoms)
        info["r_belief"] = r_bel

        if self.expose_accept_prob_in_obs and isinstance(obs, dict):
            obs = dict(obs)
            obs["accept_prob"] = ap

        # **兼容 torch_ac：返回 4 元组**
        return obs, r, done, info
