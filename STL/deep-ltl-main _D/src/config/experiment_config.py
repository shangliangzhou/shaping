from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """实验配置类"""
    name: str  # 实验名称
    env: str  # 环境名称
    num_steps: int  # 训练模型的步数
    seed: int = 0  # 随机种子
    save_dir: str = 'experiments'  # 保存结果的目录
    log_interval: int = 1  # 记录结果的日志间隔（以更新次数为单位）
    save_interval: int = 2  # 保存模型的间隔（以更新次数为单位）
    # eval_interval: int = 200_000  # 用于评估的模型保存间隔（以episode为单位）
    eval_interval: int = 200_00  # 用于评估的模型保存间隔（以episode为单位）
    # eval_interval: int = 20 # 用于评估的模型保存间隔（以episode为单位）
    num_procs: int = 1  # 使用的进程数量
    device: str = 'cuda'  # 训练使用的设备
    ltl_sampler: str | None = None  # LTL采样器的名称

