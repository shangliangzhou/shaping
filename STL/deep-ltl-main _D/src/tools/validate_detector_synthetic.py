# tools/validate_detector_synthetic.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 添加src目录到Python路径

from perception.detector_model import PropositionDetector

det = PropositionDetector(ema_tau=0.9, temperature=1.0)
seq = [0]*10 + [1]*10 + [0]*5 + [1]*5   # 人为的命题“真/假”序列
obs = {"propositions": {"blue": 0}}
det.reset()
print("t, raw, p_hat")
for t, v in enumerate(seq):
    obs["propositions"]["blue"] = v
    p_hat = det.predict_proba(obs)["blue"]
    print(f"{t:02d}, {v}, {p_hat:.3f}")
# 预期：p_hat 不是跳变，而是逐步上升/下降（EMA 平滑）
