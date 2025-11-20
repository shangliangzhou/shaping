import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "_plot_deps"))

import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import trange
import sys
sys.path.append('/home/gh/公共/zh/deep-ltl-main/src')
from envs import make_env
from ltl import FixedSampler
from model.model import build_model
from model.agent import Agent
from config import model_configs
from sequence.search import ExhaustiveSearch
from utils.model_store import ModelStore
from visualize.zones import draw_trajectories

# 做什么：在 FlatWorld/ZoneEnv 上，把 simulate 的 rollout 画成图（障碍、目标、命题区、Agent 路径），用于论文/报告的定性展示。

# 关系：共享 simulate 的载模与执行逻辑，只是多了作图与保存 PNG/PDF。


env_name = 'PointLtl2-v0'
exp = 'zones_paper_s1'
seed = 1

random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

# sampler = FixedSampler.partial('GF blue & GF yellow')
sampler = FixedSampler.partial('GF blue & GF green & GF yellow & G !magenta')
deterministic = True

env = make_env(env_name, sampler, render_mode=None, max_steps=1000)
config = model_configs[env_name]
model_store = ModelStore(env_name, exp, seed)
model_store.load_vocab()
training_status = model_store.load_training_status(map_location='cpu')
model = build_model(env, training_status, config)

props = set(env.get_propositions())
search = ExhaustiveSearch(model, props, num_loops=2)
agent = Agent(model, search=search, propositions=props, verbose=False)

num_episodes = 4

trajectories = []
zone_poss = []

pbar = trange(num_episodes)
for i in pbar:
    env.load_world_info(f'eval_datasets/PointLtl2-v0/worlds/world_info_{i}.pkl')
    obs, info = env.reset(), {}
    agent.reset()
    done = False

    zone_poss.append(env.zone_positions)
    agent_traj = []

    while not done:
        action = agent.get_action(obs, info, deterministic=deterministic)
        action = action.flatten()
        obs, reward, done, info = env.step(action)
        agent_traj.append(env.agent_pos[:2])
        if done:
            trajectories.append(agent_traj)

env.close()
cols = 4 if len(zone_poss) > 4 else len(zone_poss)
rows = 1 if len(zone_poss) <= 4 else 2
fig = draw_trajectories(zone_poss, trajectories, cols, rows)
plt.show()
