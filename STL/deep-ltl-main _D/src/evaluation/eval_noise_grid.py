# tools/eval_noise_grid.py
import itertools, json, numpy as np
from envs import make_env
from model.model import build_model
from utils.model_store import ModelStore
import torch

def eval_once(model, env, episodes=20):
    sr = 0
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                a = model.act(obs)   # 你项目里若是 Agent/Policy 类，请替换成对应接口
            obs, r, done, info = env.step(a)
        # 成功判据：按你项目的 info 键名替换（示例）
        succ = info.get("success", info.get("goal_reached", False))
        sr += 1 if succ else 0
    return sr / episodes

def main(run_name, seeds=(1,), noise_miss=(0.0,0.05,0.1), noise_false=(0.0,0.02,0.05), delay=(0,1,2)):
    # 载入模型（按你现有目录结构，这段可能需要小调整）
    ms = ModelStore.from_name(run_name, seed=seeds[0])
    ts = ms.load_training_status()
    env = make_env("PointLtl2-v0", sampler=None, sequence=True,
                   noise_enable=True, shaping_enable=False, belief_enable=False)  # 先占位
    model = build_model(env, ts, ...)  # 按你的 build_model 接口补齐参数
    model.load_state_dict(ts["model_state"])
    model.eval()

    results = []
    for pm, pf, dl in itertools.product(noise_miss, noise_false, delay):
        env = make_env("PointLtl2-v0", sampler=None, sequence=True,
                       noise_enable=True, noise_p_miss=pm, noise_p_false=pf, noise_delay_steps=dl,
                       shaping_enable=True, shaping_beta=0.01, shaping_eta=0.02, shaping_alpha=0.5,
                       belief_enable=False)  # 或 True，评对照
        sr = eval_once(model, env, episodes=50)
        results.append({"p_miss": pm, "p_false": pf, "delay": dl, "SR": sr})
        print(results[-1])
    print("JSON:", json.dumps(results, indent=2))

if __name__ == "__main__":
    main(run_name="ppo_belief_eta002")
