# src/envs/episode_buffer_wrapper.py
from __future__ import annotations
import gymnasium as gym
from typing import Any, Dict, List, Tuple

class EpisodeBufferWrapper(gym.Wrapper):
    """
    记录原始 obs/action/done（以及 wrapper 链输出给 agent 的 reward），
    兼容底层 Gym/Gymnasium，但对外统一成【旧 Gym 接口】：
      - reset() -> obs
      - step(a) -> (obs, reward, done, info)
    对内用于 HER 的观测一律规范化为 dict（含 features/goal/initial_goal/propositions）。
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._episodes: List[Dict[str, Any]] = []
        self._cur_traj: List[Dict[str, Any]] = []
        self._last_obs_raw: Any = None   # 给上层（PPO/同步env）用
        self._last_obs_rec: Any = None   # 我们记录的规范化版本（dict）

    # ------- 规范化：从 tuple/list 里找出真正的 dict obs -------
    @staticmethod
    def _ensure_obs_dict(o: Any) -> Any:
        if isinstance(o, dict):
            return o
        if isinstance(o, (tuple, list)):
            # 广度优先在嵌套里找 dict，优先包含关键键的
            queue = list(o)
            while queue:
                cand = queue.pop(0)
                if isinstance(cand, dict):
                    ks = set(cand.keys())
                    if {"features", "goal", "initial_goal", "propositions"} & ks:
                        return cand
                elif isinstance(cand, (tuple, list)):
                    queue.extend(list(cand))
            # 兜底：第一个 dict 也收下
            for cand in o:
                if isinstance(cand, dict):
                    return cand
        return o  # 实在没有就原样返回

    # ------- 兼容性解包（对内使用） -------
    @staticmethod
    def _unpack_reset(ret) -> Tuple[Any, Dict[str, Any]]:
        # 兼容 Gym/Gymnasium，返回 (obs, info)
        if isinstance(ret, tuple):
            if len(ret) == 0:
                raise ValueError("env.reset() returned empty tuple")
            obs = ret[0]
            info = ret[1] if len(ret) > 1 and isinstance(ret[1], dict) else {}
            return obs, info
        return ret, {}

    @staticmethod
    def _unpack_step(ret) -> Tuple[Any, float, bool, Dict[str, Any]]:
        # 兼容 4元 or 5元，统一为 (obs, reward, done, info)
        if not isinstance(ret, tuple):
            raise ValueError("env.step() must return a tuple")
        n = len(ret)
        if n == 5:
            next_obs, reward, terminated, truncated, info = ret
            done = bool(terminated or truncated)
            return next_obs, float(reward), done, info
        if n == 4:
            next_obs, reward, done, info = ret
            return next_obs, float(reward), bool(done), info
        # 非常规：尽量取前四个
        next_obs = ret[0]
        reward   = float(ret[1]) if n > 1 else 0.0
        done     = bool(ret[2])  if n > 2 else False
        info     = ret[3]        if n > 3 and isinstance(ret[3], dict) else {}
        return next_obs, reward, done, info

    # ------- 覆写 reset/step（对外统一旧 Gym 接口） -------
    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        obs_raw, info = self._unpack_reset(ret)

        # 规范化给我们自己记录
        obs_rec = self._ensure_obs_dict(obs_raw)

        self._flush_if_unfinished()
        self._cur_traj = []
        self._last_obs_raw = obs_raw   # 对外
        self._last_obs_rec = obs_rec   # 对内（HER）
        return obs_raw                  # <<< 对外只返回 obs（旧Gym）

    def step(self, action):
        prev_obs_rec = self._last_obs_rec

        ret = self.env.step(action)
        next_obs_raw, reward, done, info = self._unpack_step(ret)
        next_obs_rec = self._ensure_obs_dict(next_obs_raw)

        # 记录（用规范化后的 dict）
        self._cur_traj.append({
            "obs": prev_obs_rec,
            "action": action,
            "reward": reward,   # 这是 wrapper 链最终给 PPO 的奖励（可能已形塑）
            "done": done,
            "info": info,
        })
        self._last_obs_raw = next_obs_raw
        self._last_obs_rec = next_obs_rec

        if done:
            obss  = [t["obs"]    for t in self._cur_traj]
            acts  = [t["action"] for t in self._cur_traj]
            rews  = [t["reward"] for t in self._cur_traj]
            dones = [False] * (len(self._cur_traj) - 1) + [True]
            self._episodes.append({"obss": obss, "acts": acts, "rews_shaped": rews, "dones": dones})
            self._cur_traj = []

        # 对外统一旧 Gym：4 元组
        return next_obs_raw, reward, done, info

    # ------- episode 管理 -------
    def _flush_if_unfinished(self):
        if len(self._cur_traj) > 0:
            obss  = [t["obs"]    for t in self._cur_traj]
            acts  = [t["action"] for t in self._cur_traj]
            rews  = [t["reward"] for t in self._cur_traj]
            dones = [False] * len(self._cur_traj)
            self._episodes.append({"obss": obss, "acts": acts, "rews_shaped": rews, "dones": dones})
            self._cur_traj = []

    def pop_recent_episodes(self) -> List[Dict[str, Any]]:
        eps = self._episodes
        self._episodes = []
        return eps
