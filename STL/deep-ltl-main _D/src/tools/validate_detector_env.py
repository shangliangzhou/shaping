# tools/validate_detector_env.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 添加src目录到Python路径
from perception.detector_model import PropositionDetector
import numpy as np
from envs import make_env
from sequence.samplers import CurriculumSampler, curricula

env = make_env("PointLtl2-v0", CurriculumSampler.partial(curricula["PointLtl2-v0"]), sequence=True)
det = PropositionDetector(ema_tau=0.9, temperature=1.0)
det.reset()
# 处理 gym/gymnasium API 的兼容性
reset_result = env.reset()
if isinstance(reset_result, tuple):
    obs, info = reset_result
else:
    obs = reset_result
    info = {}

T = 200
recs = []
for t in range(T):
    # 随机动作（只是采样验证概率，不做训练）
    a = env.action_space.sample()
    step_result = env.step(a)
    
    # 处理 step 返回值的兼容性
    if len(step_result) == 5:  # gymnasium 风格 (obs, reward, terminated, truncated, info)
        obs, r, terminated, truncated, info = step_result
        done = terminated or truncated
    else:  # gym 风格 (obs, reward, done, info)
        obs, r, done, info = step_result
    
    # propositions 是一个 set，包含当前为真的命题
    raw_props = obs.get("propositions", set())
    p_hat = det.predict_proba(obs)
    # 检查 'blue' 是否在当前活跃的命题集合中
    v_raw = float('blue' in raw_props)
    v_hat = float(p_hat.get("blue", 0.5))
    recs.append((v_raw, v_hat))
    if done:
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}

arr = np.array(recs)  # (T, 2) → [raw, p_hat]
brier = ((arr[:,1] - arr[:,0])**2).mean()
print("Brier score(blue) =", brier)
# 预期：p_hat ∈ [0,1]，且在“蓝命题”长时间为真/假时，p_hat 会接近 1/0。
