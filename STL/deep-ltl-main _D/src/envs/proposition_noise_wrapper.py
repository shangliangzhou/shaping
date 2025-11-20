# src/envs/proposition_noise_wrapper.py
from __future__ import annotations
try:
    import gymnasium as gym
except Exception:
    import gym

from typing import Dict, Any, Union
import numpy as np
ObsType = Union[Dict[str, Any], Any]

class PropositionNoiseWrapper(gym.Wrapper):
    """
    对 obs['propositions'] 或 obs['prop_probs'] 注入噪声：
      - p_miss:     真→假 的漏检概率
      - p_false:    假→真 的误报概率
      - delay_steps: 命题延迟（FIFO 缓冲），对布尔与概率都生效
    支持 per-atom 参数（传 dict），否则用标量。
    """
    def __init__(self,
                 env: gym.Env,
                 p_miss=0.0,
                 p_false=0.0,
                 delay_steps: int = 0,
                 seed: int | None = None):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)
        self.p_miss = p_miss
        self.p_false = p_false
        self.delay_steps = int(delay_steps)
        self._fifo = {}  # {atom: list} 延迟缓冲

    def _get_prob(self, name: str, param, default=0.0) -> float:
        if isinstance(param, dict):
            return float(param.get(name, default))
        return float(param)

    def _apply_noise_bool(self, props: Dict[str, int]) -> Dict[str, int]:
        noisy = {}
        for k, v in props.items():
            v = 1 if v else 0
            pm = self._get_prob(k, self.p_miss, 0.0)
            pf = self._get_prob(k, self.p_false, 0.0)
            if v == 1 and self.rng.random() < pm:
                v = 0
            elif v == 0 and self.rng.random() < pf:
                v = 1
            noisy[k] = v
        return noisy

    def _apply_noise_prob(self, probs: Dict[str, float]) -> Dict[str, float]:
        noisy = {}
        for k, p in probs.items():
            p = float(p)
            pm = self._get_prob(k, self.p_miss, 0.0)
            pf = self._get_prob(k, self.p_false, 0.0)
            # 简单“信道模型”：与“错误标签”按概率混合
            p = (1 - pm - pf) * p + pf * 1.0 + pm * 0.0
            p = min(max(p, 0.0), 1.0)
            noisy[k] = p
        return noisy

    def _apply_delay(self, vals: Dict[str, Any]) -> Dict[str, Any]:
        if self.delay_steps <= 0:
            return vals
        out = {}
        for k, v in vals.items():
            buf = self._fifo.setdefault(k, [])
            buf.append(v)
            if len(buf) <= self.delay_steps:
                out[k] = buf[0]  # 初期就把最早的复用
            else:
                out[k] = buf[-1 - self.delay_steps]
                # 控制长度，避免无限增长
                if len(buf) > self.delay_steps + 64:
                    self._fifo[k] = buf[-(self.delay_steps + 8):]
        return out

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        self._fifo.clear()

        if isinstance(obs, dict):
            if "prop_probs" in obs and isinstance(obs["prop_probs"], dict):
                probs = self._apply_noise_prob(obs["prop_probs"])
                probs = self._apply_delay(probs)
                obs = dict(obs); obs["prop_probs"] = probs
            elif "propositions" in obs and isinstance(obs["propositions"], dict):
                props = self._apply_noise_bool(obs["propositions"])
                props = self._apply_delay(props)
                obs = dict(obs); obs["propositions"] = props

        if isinstance(out, tuple) and len(out) == 2:
            return obs, info
        return obs

    def step(self, action):
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            obs, r, term, trunc, info = out
            done = bool(term or trunc)
        elif isinstance(out, tuple) and len(out) == 4:
            obs, r, done, info = out
        else:
            raise RuntimeError("Unexpected env.step() return format")

        if isinstance(obs, dict):
            if "prop_probs" in obs and isinstance(obs["prop_probs"], dict):
                probs = self._apply_noise_prob(obs["prop_probs"])
                probs = self._apply_delay(probs)
                obs = dict(obs); obs["prop_probs"] = probs
            elif "propositions" in obs and isinstance(obs["propositions"], dict):
                props = self._apply_noise_bool(obs["propositions"])
                props = self._apply_delay(props)
                obs = dict(obs); obs["propositions"] = props

        return obs, float(r), bool(done), dict(info)
