# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
import math
import gymnasium as gym

DEFAULT_COLORS = ["blue", "green", "yellow", "magenta"]

@dataclass
class BeliefParams:
    # sdist -> prob 的温度化参数
    k: float = 5.0           # 斜率，越大越接近硬门
    m: float = 0.0           # 安全边距（sdist - m）
    # 时序平滑
    ema_lambda: float = 0.8  # EMA 系数
    # 稳定门（滞回 + 连帧）
    theta_up: float = 0.8
    theta_down: float = 0.2
    k_stab: int = 3          # 连续几帧触发

@dataclass
class _BeliefState:
    b: float = 0.0           # EMA 后的置信
    cnt_up: int = 0
    cnt_down: int = 0
    label: bool = False      # 当前稳健布尔

class BeliefFilter:
    def __init__(self, colors: List[str], params: BeliefParams):
        self.colors = list(colors)
        self.params = params
        self.state: Dict[str, _BeliefState] = {c: _BeliefState() for c in self.colors}

    def _sigmoid(self, x: float) -> float:
        # 数值安全的 Sigmoid
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)

    def update(self, sdist: Dict[str, float]) -> tuple[Dict[str, float], Dict[str, float], Set[str]]:
        """给定每色 sdist，输出 (p_atoms, b_atoms, L_stab)"""
        p_atoms: Dict[str, float] = {}
        b_atoms: Dict[str, float] = {}
        L_stab: Set[str] = set()

        k, m = self.params.k, self.params.m
        lam = self.params.ema_lambda
        th_up, th_dn = self.params.theta_up, self.params.theta_down
        k_stab = self.params.k_stab

        for c in self.colors:
            s = float(sdist.get(f"sdist_{c}", 0.0))
            p = self._sigmoid(k * (s - m))
            st = self.state[c]
            # EMA
            st.b = lam * st.b + (1.0 - lam) * p

            # 稳定门（双阈 + 连帧）
            if st.b >= th_up:
                st.cnt_up += 1
                st.cnt_down = 0
                if st.cnt_up >= k_stab:
                    st.label = True
            elif st.b <= th_dn:
                st.cnt_down += 1
                st.cnt_up = 0
                if st.cnt_down >= k_stab:
                    st.label = False
            else:
                # 中间带：不改变 label，但计数清零，避免误触发
                st.cnt_up = 0
                st.cnt_down = 0

            p_atoms[c] = p
            b_atoms[c] = st.b
            if st.label:
                L_stab.add(c)

        return p_atoms, b_atoms, L_stab

class BeliefWrapper(gym.Wrapper):
    """
    从 info['sdist_*'] 构造命题信念与稳健标签:
      - info['p_atoms'] : 每色瞬时概率
      - info['b_atoms'] : 每色 EMA 概率
      - info['L_stab']  : 稳健命题集合（供 LDBA 推进）
      - info['proposition_source'] = 'stable'
    依赖：上游 wrapper 已提供每步 info['sdist_<color>']。
    """
    def __init__(
        self,
        env: gym.Env,
        colors: Optional[List[str]] = None,
        params: Optional[BeliefParams] = None
    ):
        super().__init__(env)
        if colors is None:
            # 优先取上游 wrapper 的颜色列表
            colors = getattr(env, "colors", None) or DEFAULT_COLORS
        self.colors = list(colors)
        self.params = params or BeliefParams()
        self.filter = BeliefFilter(self.colors, self.params)

    def _build_sdist_dict(self, info: Dict[str, Any]) -> Dict[str, float]:
        return {f"sdist_{c}": float(info.get(f"sdist_{c}", 0.0)) for c in self.colors}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        sdist = self._build_sdist_dict(info)
        p_atoms, b_atoms, L_stab = self.filter.update(sdist)
        info['p_atoms'] = p_atoms
        info['b_atoms'] = b_atoms
        info['L_stab'] = L_stab
        info['proposition_source'] = 'stable'
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        sdist = self._build_sdist_dict(info)
        p_atoms, b_atoms, L_stab = self.filter.update(sdist)
        info['p_atoms'] = p_atoms
        info['b_atoms'] = b_atoms
        info['L_stab'] = L_stab
        info['proposition_source'] = 'stable'
        return obs, reward, terminated, truncated, info
