# src/envs/proposition_noise_wrapper.py
from __future__ import annotations
try:
    import gymnasium as gym
except Exception:
    import gym

import random
from typing import Optional, Iterable, Set, List

class PropositionNoiseWrapper(gym.Wrapper):
    """
    评测端命题噪声（集合型）：
      - 漏报：以 p_miss 概率删掉原本为真的命题
      - 误报：对全集内为假的每个命题，以 p_false 概率加入
    仅改 info['propositions']（集合[str]），不改 obs/reward。
    额外：支持互斥组（mutex_groups），用于强制“一热”，避免 LDBA 无出边。
    """
    def __init__(self, env,
                 ap_list: Optional[Iterable[str]] = None,
                 p_miss: float = 0.0,
                 p_false: float = 0.0,
                 seed: Optional[int] = None,
                 mutex_groups: Optional[List[Set[str]]] = None):
        super().__init__(env)
        assert 0.0 <= p_miss <= 1.0 and 0.0 <= p_false <= 1.0
        self.p_miss = float(p_miss)
        self.p_false = float(p_false)
        self.rng = random.Random(seed)

        self.ap_universe: Set[str] = set(ap_list) if ap_list else set()
        self._seen_aps: Set[str] = set()

        # 互斥组：例如 [{blue, green, yellow, magenta}]
        self.mutex_groups: List[Set[str]] = []
        if mutex_groups:
            self.mutex_groups = [set(g) for g in mutex_groups]

        self.stats = {"steps": 0, "miss_del": 0, "false_add": 0}

    def reset(self, **kwargs):
        self.stats.update({"steps": 0, "miss_del": 0, "false_add": 0})
        out = self.env.reset(**kwargs)
        obs, info = (out if isinstance(out, tuple) and len(out) == 2 else (out, {}))
        return obs, self._maybe_noise(info)

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            return obs, reward, terminated, truncated, self._maybe_noise(info)
        else:
            obs, reward, done, info = out
            return obs, reward, done, self._maybe_noise(info)

    def _apply_noise_once(self, true_set: Set[str]) -> Set[str]:
        # 更新 AP 全集
        self._seen_aps.update(true_set)
        if not self.ap_universe:
            self.ap_universe = set(self._seen_aps)

        noisy = set()
        # 漏报
        for p in true_set:
            if self.rng.random() >= self.p_miss:
                noisy.add(p)
            else:
                self.stats["miss_del"] += 1

        # 误报
        for p in self.ap_universe:
            if p not in noisy and self.rng.random() < self.p_false:
                noisy.add(p)
                self.stats["false_add"] += 1

        return noisy

    def _enforce_mutex_groups(self, s: Set[str]) -> Set[str]:
        """对互斥组强制一热：若组内多于1个为真，只保留1个。"""
        if not self.mutex_groups:
            return s
        s = set(s)  # copy
        for group in self.mutex_groups:
            inter = s & group
            if len(inter) > 1:
                keep = self.rng.choice(tuple(inter))
                s -= inter
                s.add(keep)
        return s

    def _maybe_noise(self, info: dict) -> dict:
        props = info.get("propositions", None)
        if props is None:
            return info
        if not isinstance(props, (set, frozenset)):
            # 如果已被上游布尔化成 {ap: bool}，我们不处理
            return info

        self.stats["steps"] += 1
        true_set = set(props)

        noisy = self._apply_noise_once(true_set)
        noisy = self._enforce_mutex_groups(noisy)

        info["propositions_raw"] = true_set       # 调试
        info["propositions_noisy"] = set(noisy)   # 调试
        info["propositions"] = noisy              # 交给 LDBA
        info["noise_stats"] = dict(self.stats)
        return info
