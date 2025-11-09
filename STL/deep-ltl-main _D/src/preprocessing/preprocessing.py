from typing import Any

import torch

import torch_ac
import numpy as np

from ltl.automata import LDBASequence
from ltl.logic import FrozenAssignment, Assignment
from preprocessing.vocab import VOCAB
from preprocessing.batched_sequences import BatchedReachAvoidSequences, ReachAvoidSet


def preprocess_obss(obss: list[dict[str, Any]], propositions: set[str], device=None) -> torch_ac.DictList:
    """
    预处理观测数据，将其转换为模型训练所需的格式。
    
    该函数主要完成以下任务：
    1. 提取观测中的特征和目标序列
    2. 计算epsilon转移的掩码
    3. 将数据打包成torch_ac.DictList格式
    
    参数:
        obss: 观测数据列表，每个元素是一个字典obs，包含"features"和"goal"等键
        propositions: 命题集合，用于构建赋值映射
        device: 指定计算设备，默认为None
        
    返回:
        torch_ac.DictList: 包含预处理后特征、序列和epsilon掩码的字典列表
    """
    features = []
    seqs = []
    epsilon_mask = []
    #遍历每个观测，提取特征和目标序列
    for obs in obss:
        obs = _to_obs_dict(obs)  # <<< 新增：先归一化为 dict >>>
        features.append(obs["features"])
        #seq是（reach,avoid）组成的序列，seq[-1][0]表示最后一个reach集合(即可能出现epsilon转移的集合)
        seqs.append(list(reversed(obs["goal"])))
    #计算每个序列的epsilon转移掩码
    for seq, obs in zip(seqs, obss):
        #获取epsilon转移的索引
        epsilon_enabled = seq[-1][0] == LDBASequence.EPSILON
        if epsilon_enabled and len(seq) > 1:
            next_avoid = seq[-2][1]
            assignment = Assignment({p: (p in obs['propositions']) for p in propositions}).to_frozen()
            #验证epsilon转移是否被允许
            epsilon_enabled &= assignment not in next_avoid
        epsilon_mask.append(epsilon_enabled)
    return torch_ac.DictList({
        "features": preprocess_features(features, device=device), #
        "seq": BatchedReachAvoidSequences([preprocess_sequence(seq) for seq in seqs], device=device),
        "epsilon_mask": torch.tensor(epsilon_mask, dtype=torch.bool).to(device),
    })

def _to_obs_dict(o):
    # 已是 dict
    if isinstance(o, dict):
        return o
    # 从 tuple/list 里找嵌套 dict（优先含关键键）
    if isinstance(o, (tuple, list)):
        queue = list(o)
        while queue:
            x = queue.pop(0)
            if isinstance(x, dict):
                ks = set(x.keys())
                if {"features", "goal", "initial_goal", "propositions"} & ks:
                    return x
            elif isinstance(x, (tuple, list)):
                queue.extend(list(x))
        # 兜底：第一个 dict 也返回
        for x in o:
            if isinstance(x, dict):
                return x
    return o  # 实在没有就原样交给后续（一般会报清晰错误）

# 将features转换为numpy数组
# 创建float类型的PyTorch张量
# 将张量移动到指定设备（如GPU） 主要用途是数据预处理，为神经网络训练准备张量格式的数据。
def preprocess_features(features, device=None) -> torch.tensor:
    return torch.tensor(np.array(features), dtype=torch.float).to(device)


def preprocess_sequence(seq: LDBASequence) -> list[ReachAvoidSet]:
    #对序列进行预处理，将每个元素中的两个部分分别进行赋值预处理
    return [(preprocess_assignments(a), preprocess_assignments(b)) for a, b in seq]


def preprocess_assignments(assignments: frozenset[FrozenAssignment] | type(LDBASequence.EPSILON)) -> list[int]:
    
    if assignments == LDBASequence.EPSILON:
        return [VOCAB['EPSILON']]
    if len(assignments) == 0:
        return [VOCAB['NULL']]
    return [VOCAB[a] for a in assignments]
