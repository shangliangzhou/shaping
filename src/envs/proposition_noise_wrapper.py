# src/envs/proposition_noise_wrapper.py
from __future__ import annotations
from typing import Any, Iterable
import gymnasium
import numpy as np


class PropositionNoiseWrapper(gymnasium.Wrapper):
    """
    评测期命题噪声注入（只作用于 *原始命题* info['propositions']）：
      - 不再使用/修改 L_stab 或 propositions_geom，确保我们的稳定门仍然保持“干净”。
      - p_miss : 把真命题以该概率丢弃（漏检）
      - p_false: 从非真命题里以该概率加入假阳性（误报）
      - 可选互斥组 mutex_groups：如 {'blue','green','yellow','magenta'}，限制同组最多出现一个。
    附带最小统计:
      info['propositions_raw']   : 注入前的原始集合（每步覆盖）
      info['propositions_noisy'] : 注入后的集合（与 info['propositions'] 相同）
      info['noise_stats']        : {'steps', 'miss_del', 'false_add'}
      info['noise_p_miss'], info['noise_p_false'], info['noise_source'] = 'propositions'
    """

    def __init__(
        self,
        env: gymnasium.Env,
        ap_list: Iterable[str] | None = None,
        p_miss: float = 0.0,
        p_false: float = 0.0,
        seed: int | None = None,
        mutex_groups: list[set[str]] | None = None,
    ):
        super().__init__(env)
        self.p_miss = float(p_miss)
        self.p_false = float(p_false)
        self.rng = np.random.RandomState(None if seed is None else int(seed))

        # 命题全集（用于误报采样）
        if ap_list is None:
            if hasattr(env, "get_propositions"):
                ap_list = list(sorted(env.get_propositions()))
            else:
                ap_list = []
        self.ap_list = list(ap_list)

        # 互斥组（可为空）
        self.mutex_groups = mutex_groups or []

        # 统计计数
        self._steps = 0
        self._miss_del = 0
        self._false_add = 0

    # ---- 工具：互斥兜底（若同组出现多个，仅保留第一个）----
    def _apply_mutex(self, s: set[str]) -> set[str]:
        for g in self.mutex_groups:
            present = [a for a in s if a in g]
            if len(present) > 1:
                keep = present[0]
                for a in present[1:]:
                    s.discard(a)
                s.add(keep)
        return s

    # ---- 噪声主过程：只读取/写回 info['propositions'] ----
    def _noisify(self, info: dict[str, Any]) -> None:
        raw = set(info.get("propositions", set()))
        # 1) 漏检（删除）
        kept = {a for a in raw if self.rng.rand() > self.p_miss}
        miss_del = len(raw) - len(kept)

        # 2) 误报（添加）
        #    从全集减去当前 kept 的剩余候选里，按 p_false 逐个尝试加入；
        #    若该候选所在互斥组已有元素，则跳过。
        false_add = 0
        candidates = [a for a in self.ap_list if a not in kept]
        for a in candidates:
            if self.rng.rand() < self.p_false:
                allow = True
                for g in self.mutex_groups:
                    if a in g and any(b in kept for b in g):
                        allow = False
                        break
                if allow:
                    kept.add(a)
                    false_add += 1

        # 3) 互斥兜底（极端情况下保证同组唯一）
        kept = self._apply_mutex(kept)

        # 统计 & 回写（**只**覆盖原始命题）
        self._steps += 1
        self._miss_del += miss_del
        self._false_add += false_add

        info["propositions_raw"] = raw
        info["propositions"] = kept
        info["propositions_noisy"] = set(kept)  # 方便对照查看
        info["noise_source"] = "propositions"
        info["noise_p_miss"] = self.p_miss
        info["noise_p_false"] = self.p_false
        info["noise_stats"] = {
            "steps": self._steps,
            "miss_del": self._miss_del,
            "false_add": self._false_add,
        }

    # ---- 接口适配：reset/step 兼容 4/5 元组 ----
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        ret = super().reset(seed=seed, options=options)
        if isinstance(ret, tuple) and len(ret) == 2:
            obs, info = ret
            self._noisify(info)
            return obs, info
        # 极少数环境可能只返回 obs
        return ret

    def step(self, action):
        ret = super().step(action)

        if not isinstance(ret, tuple):
            raise RuntimeError("env.step must return a tuple")

        if len(ret) == 5:
            obs, rew, terminated, truncated, info = ret
            self._noisify(info)
            return obs, rew, terminated, truncated, info
        elif len(ret) == 4:
            obs, rew, done, info = ret
            self._noisify(info)
            # 维持 gymnasium 5 元组风格向上游传播
            terminated, truncated = bool(done), False
            return obs, rew, terminated, truncated, info
        else:
            raise RuntimeError(f"Unsupported step return length: {len(ret)}")
