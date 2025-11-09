# src/envs/potential_shaping_wrapper.py
from __future__ import annotations

from typing import Any, Dict, Optional, Iterable, Tuple, Hashable, Set
import math

try:
    import gymnasium as gym
except Exception:
    import gym


class PotentialShapingWrapper(gym.Wrapper):
    r"""
    Training-only potential-based reward shaping for sequence goals.

    Phi(s) = - ( remain(s) + alpha * d_goal(s) )
      - remain(s) = len(obs['goal'])
      - d_goal(s) = 1 - |P ∩ G| / |G|,   P: active propositions from obs['propositions'],
                                         G: required AP set of the NEXT subgoal (obs['goal'][0])

    Shaping:
      r' = r + beta * ( gamma * Phi(s') - Phi(s) ) + eta * 1{remain(s') < remain(s)}

    Works with obs:
      - 'goal'          : remaining goal sequence (list/tuple; next subgoal = goal[0])
      - 'initial_goal'  : full goal sequence (unused but kept for future)
      - 'propositions'  : active propositions (set/list/tuple/dict{ap:bool})
      - 'features'      : raw env obs (unused here)

    Notes:
      * DO NOT wrap env for evaluation/inference; this is training-time only.
      * If your subgoal element format differs, adapt `_to_ap_set(next_goal)` accordingly.
    """

    def __init__(
        self,
        env: gym.Env,
        beta: float,
        eta: float,
        gamma: float,
        alpha: float = 0.0,   # weight of d_goal(s) term
    ):
        super().__init__(env)
        self.beta = float(beta)
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

        # memory of previous state's components for Phi
        self._last_remain: Optional[int] = None
        self._last_dgoal: Optional[float] = None

        # one-time prints
        self._printed_once = False

        # diagnostics
        self._shape_sum = 0.0
        self._prog_hits = 0

    # ---------------- reset / step ----------------

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        obs = out[0] if isinstance(out, tuple) else out

        remain0 = self._remain_len(obs)
        dgoal0  = self._d_goal(obs)
        self._last_remain = remain0
        self._last_dgoal  = dgoal0

        if not self._printed_once:
            print(
                f"[PotentialShapingWrapper] activated: "
                f"beta={self.beta}, eta={self.eta}, gamma={self.gamma}, alpha={self.alpha}, "
                f"remain0={remain0}, dgoal0={dgoal0}"
            )
            self._printed_once = True

        return out  # keep original arity/order

    def step(self, action):
        result = self.env.step(action)
        if not isinstance(result, tuple):
            return result

        L = len(result)
        if L == 5:      # Gymnasium
            obs_next, reward, terminated, truncated, info = result
            done_out = None
        elif L == 4:    # Gym
            obs_next, reward, done, info = result
            terminated, truncated = done, False
            done_out = done
        else:
            out = list(result)
            try:
                obs_next = out[0]; reward = out[1]
            except Exception:
                return result
            terminated = truncated = None
            info = out[-1] if isinstance(out[-1], dict) else {}
            done_out = None

        # --- compute shaping ---
        shaping = 0.0

        remain_next = self._remain_len(obs_next)
        dgoal_next  = self._d_goal(obs_next)

        # build Phi_prev/Phi_next if we have previous memory
        if remain_next is not None:
            remain_prev = self._last_remain if self._last_remain is not None else remain_next
            dgoal_prev  = self._last_dgoal  if self._last_dgoal  is not None else dgoal_next

            phi_prev = float(remain_prev) + self.alpha * float(dgoal_prev or 0.0)
            phi_next = float(remain_next) + self.alpha * float(dgoal_next or 0.0)

            # Phi(s) = -(...)  => shaping = beta * (phi_prev - gamma*phi_next)
            shaping += self.beta * (phi_prev - self.gamma * phi_next)

            # tiny pulse when we strictly reduce remain (progress to next subgoal)
            if self.eta > 0.0 and (self._last_remain is not None) and (remain_next < self._last_remain):
                shaping += self.eta
                self._prog_hits += 1

        # update memory
        self._last_remain = remain_next if remain_next is not None else self._last_remain
        self._last_dgoal  = dgoal_next  if dgoal_next  is not None else self._last_dgoal

        # apply shaping
        reward = float(reward) + float(shaping)
        self._shape_sum += float(shaping)

        # return with original arity/order
        if L == 5:
            return obs_next, reward, terminated, truncated, info
        elif L == 4:
            return obs_next, reward, done_out, info
        else:
            out[1] = reward
            return tuple(out)

    # ---------------- diagnostics for logger ----------------

    def get_shaping_stats_and_reset(self) -> Dict[str, float]:
        out = {"step_shaping_sum": float(self._shape_sum),
               "progress_hits": float(self._prog_hits)}
        self._shape_sum = 0.0
        self._prog_hits = 0
        return out

    # ---------------- helpers: remain & d_goal ----------------

    def _remain_len(self, obs: Any) -> Optional[int]:
        """len(obs['goal']) if available; else None."""
        if isinstance(obs, dict) and "goal" in obs:
            g = obs["goal"]
            if isinstance(g, (list, tuple)):
                return len(g)
            # 某些实现里 'goal' 可能是可迭代的自定义类型
            try:
                return len(list(g))
            except Exception:
                return None
        return None

    def _d_goal(self, obs: Any) -> Optional[float]:
        """
        计算到“下一子目标”的匹配距离 d_goal ∈ [0,1]：
          d = 1 - |P ∩ G| / |G|
        - 若剩余序列为空（G 为空），返回 0（已完成）。
        - 若取不到 G 或 P，返回 None（本步不加几何项）。
        """
        if not isinstance(obs, dict):
            return None
        # next goal
        goal_seq = obs.get("goal", None)
        if goal_seq is None:
            return None
        try:
            goal_seq = list(goal_seq)
        except Exception:
            return None
        if len(goal_seq) == 0:
            return 0.0  # no remaining goal

        next_goal = goal_seq[0]
        G = self._to_ap_set(next_goal)
        if G is None or len(G) == 0:
            return 0.0

        # active propositions
        P = self._to_active_ap_set(obs.get("propositions", None))
        if P is None:
            return None

        inter = len(P.intersection(G))
        d = 1.0 - float(inter) / float(len(G))
        # clamp
        if d < 0.0: d = 0.0
        if d > 1.0: d = 1.0
        return d

    # ---------------- adapters for various formats ----------------

    @staticmethod
    def _to_ap_set(x: Any) -> Optional[Set[Hashable]]:
        """
        将“下一子目标”的表示转为 AP 集合：
          - 'red' -> {'red'}
          - ['red','blue'] -> {'red','blue'}
          - {'red': True, 'blue': False} -> {'red'}
          - 自定义对象：若有 .atoms / .aps / .literals / .symbols 等属性，尝试读取
        """
        if x is None:
            return None
        # string: treat as single AP
        if isinstance(x, str):
            return {x}
        # list/tuple/set of labels
        if isinstance(x, (list, tuple, set)):
            S: Set[Hashable] = set()
            for e in x:
                if isinstance(e, str):
                    S.add(e)
                elif hasattr(e, "name"):
                    S.add(getattr(e, "name"))
                else:
                    S.add(e)
            return S
        # dict of label->bool
        if isinstance(x, dict):
            S = {k for k, v in x.items() if bool(v)}
            return S
        # object with common attributes
        for attr in ("atoms", "aps", "literals", "symbols"):
            if hasattr(x, attr):
                try:
                    vals = getattr(x, attr)
                    return set(vals) if not isinstance(vals, dict) else set(vals.keys())
                except Exception:
                    pass
        # fallback: try to stringify
        try:
            return {str(x)}
        except Exception:
            return None

    @staticmethod
    def _to_active_ap_set(p: Any) -> Optional[Set[Hashable]]:
        """
        将 obs['propositions'] 转为“当前为 True 的 AP 集合”：
          - {'red': True, 'blue': False} -> {'red'}
          - ['red','green'] / set(...)   -> 同名集合
          - 自定义对象：有 .true / .active / .positives / .atoms 等属性的，尽可能解析
        """
        if p is None:
            return None
        if isinstance(p, set):
            return set(p)
        if isinstance(p, list) or isinstance(p, tuple):
            return set(p)
        if isinstance(p, dict):
            return {k for k, v in p.items() if bool(v)}
        for attr in ("true", "active", "positives", "atoms", "aps"):
            if hasattr(p, attr):
                try:
                    vals = getattr(p, attr)
                    return set(vals) if not isinstance(vals, dict) else {k for k, v in vals.items() if bool(v)}
                except Exception:
                    pass
        # string → 单一激活 AP
        if isinstance(p, str):
            return {p}
        try:
            return set(p)  # last resort
        except Exception:
            return None
