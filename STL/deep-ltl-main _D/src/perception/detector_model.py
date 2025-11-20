# src/perception/detector_model.py
from __future__ import annotations
from typing import Dict, Any, Optional

class PropositionDetector:
    """
    轻量命题“概率化”器：
      - 若 obs 中已有 'prop_probs'（0~1），直接返回；
      - 否则读取 'propositions'（{p:0/1})，做 EMA 平滑得到概率；
      - 可选温度校准（温度 T>0，小于1变“更自信”，大于1变“更保守”）。
    """
    def __init__(self, ema_tau: float = 0.9, temperature: float = 1.0):
        assert 0.0 <= ema_tau < 1.0
        assert temperature > 0.0
        self.ema_tau = ema_tau
        self.temperature = temperature
        self._ema: Dict[str, float] = {}

    @staticmethod
    def _sigmoid(x: float) -> float:
        import math
        return 1.0 / (1.0 + math.exp(-x))

    def _temp_scale(self, p: float) -> float:
        # 将概率映射到logit上做温度缩放，再映回概率域；防止数值问题加eps。
        import math
        eps = 1e-6
        p = min(max(p, eps), 1.0 - eps)
        logit = math.log(p / (1.0 - p))
        logit /= self.temperature
        return self._sigmoid(logit)

    def reset(self):
        self._ema.clear()

    def predict_proba(self, obs: Dict[str, Any]) -> Dict[str, float]:
        # 1) 直接用 prop_probs
        if isinstance(obs, dict) and "prop_probs" in obs and isinstance(obs["prop_probs"], dict):
            probs = {k: float(v) for k, v in obs["prop_probs"].items()}
            if self.temperature != 1.0:
                probs = {k: self._temp_scale(v) for k, v in probs.items()}
            # 也用 ema 做一点平滑（可选）
            out = {}
            for k, v in probs.items():
                prev = self._ema.get(k, v)
                nv = self.ema_tau * prev + (1.0 - self.ema_tau) * v
                self._ema[k] = nv
                out[k] = nv
            return out

        # 2) 用 propositions（0/1）做概率化与 EMA
        if isinstance(obs, dict) and "propositions" in obs and isinstance(obs["propositions"], dict):
            out = {}
            for k, v in obs["propositions"].items():
                v = 1.0 if bool(v) else 0.0
                prev = self._ema.get(k, v)
                nv = self.ema_tau * prev + (1.0 - self.ema_tau) * v
                self._ema[k] = nv
                out[k] = nv if self.temperature == 1.0 else self._temp_scale(nv)
            return out

        # 3) 无可用字段，返回空
        return {}
