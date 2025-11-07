#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "_plot_deps"))

import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme(font_scale=2.2)

# ====================== 配置 ======================
ENV = "PointLtl2-v0"      # "LetterEnv-v0" / "FlatWorld-v0"
# EXPERIMENTS = ["zones_paper_s1", "zones_paper_s1_shapping","ppo_shaping_alpha05","ppo_shaping_her"]  # 注意拼写
EXPERIMENTS = ["ppo_shaping_her","ppo_wp_k03", "ppo_wp_k02"] 
NAME_MAPPING = {
    "zones_paper_s1": "Zones S1",
    "zones_paper_s1_shapping": "Zones S1 + Shaping",
    "ppo_shaping_alpha05": "PPO Shaping + alpha=0.5",
    "ppo_shaping_her": "PPO Shaping",
    "ppo_wp_k02": "PPO Waypoint + κ=0.02",
    "deepltl": "Curriculum",
    "gcrl": "GCRL-LTL",
    "ltl2action": "LTL2Action",
    "deepset": "DeepLTL",
    "deepset_complex": "DeepLTL",
    "nocurriculum": "No curriculum",
}
SEARCH_ROOTS = ["experiments/ppo", "experiments/sac", "eval_results"]

# 选择要画的指标：可选 "return", "success_rate", "violation_rate", "avg_steps"
# 如果 METRIC="return"：画 每回合平均折扣回报 随训练步数的变化（跨随机种子取均值 + 误差带）。

# 改成 METRIC="success_rate"：画 成功率 曲线。

# 改成 METRIC="violation_rate"：画 违规率 曲线。

# 改成 METRIC="avg_steps"：画 每回合步数 曲线。
METRIC = "avg_steps"  # 可选 "return", "success_rate", "violation_rate", "avg_steps"

# 平滑窗口半径（>=2 生效）
SMOOTH_RADIUS = 9

# 误差带：True=CI 90%，False=标准差
USE_CI = True

OUT_PATH = Path("/home/gh/公共/zh/image/alpha5_her/temp.pdf")
# ==================================================


def smooth(arr: np.ndarray, radius: int) -> np.ndarray:
    if radius is None or radius <= 1 or arr.size == 0:
        return arr.astype(float)
    y = np.ones(radius, dtype=float)
    z = np.ones(arr.size, dtype=float)
    return np.convolve(arr, y, "same") / np.convolve(z, y, "same")


def infer_seed_from_path(path: Path) -> int:
    """
    从文件路径的各级目录名/文件名中尽量解析一个整数作为 seed。
    例如 .../seed_3/log.csv, .../3/log.csv, .../run-12/log.csv
    解析失败返回 -1。
    """
    # 先看文件名（不太可能）
    m = re.search(r"(\d+)", path.stem)
    if m:
        return int(m.group(1))
    # 再看父目录链
    for p in [path.parent] + list(path.parents):
        m = re.search(r"(?:seed[_-]?|run[_-]?|^)(\d+)$", p.name)
        if m:
            return int(m.group(1))
        m2 = re.search(r"(\d+)", p.name)
        if m2:
            return int(m2.group(1))
    return -1


def glob_logs(env: str, experiment: str, roots: List[str]) -> List[Path]:
    """在候选根目录中递归查找该 experiment 下的 log.csv 文件。"""
    found: List[Path] = []
    for r in roots:
        base = Path(r) / env / experiment
        if base.exists():
            found += list(base.rglob("log.csv"))
    return sorted(found)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    将你的 log.csv 列名映射为一套标准列：
      num_steps -> num_steps
      return_per_episode_mean -> return
      success_per_episode_mean -> success_rate
      violation_per_episode_mean -> violation_rate
      num_steps_per_episode_mean -> average_steps
    其它列保留原状。
    """
    col_map = {
        "num_steps": "num_steps",
        "return_per_episode_mean": "return",
        "success_per_episode_mean": "success_rate",
        "violation_per_episode_mean": "violation_rate",
        "num_steps_per_episode_mean": "average_steps",
    }
    # 仅对存在的列进行重命名
    to_rename = {k: v for k, v in col_map.items() if k in df.columns}
    return df.rename(columns=to_rename)


def process_eval_results(
    env: str,
    experiments: List[str],
    name_mapping: Optional[Dict[str, str]] = None,
    smooth_radius: int = 9,
) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []

    for exp in experiments:
        logs = glob_logs(env, exp, SEARCH_ROOTS)
        human = (name_mapping or {}).get(exp, exp)

        if not logs:
            print(f"\033[33m[WARN]\033[0m 未找到 log.csv：{human} ({exp}) 于 {', '.join(SEARCH_ROOTS)}")
            if "shapping" in exp:
                print("\033[33m[HINT]\033[0m 是否应为 'shaping'? 请核对实际目录名。")
            continue

        print(f"Loaded {len(logs)} logs for {human}")

        for fpath in logs:
            try:
                raw = pd.read_csv(fpath)
            except Exception as e:
                print(f"\033[33m[WARN]\033[0m 读取失败，跳过 {fpath}，原因：{e}")
                continue

            df = normalize_columns(raw)

            # 必须有 num_steps 和至少一个用于绘图的指标列
            if "num_steps" not in df.columns:
                print(f"\033[33m[WARN]\033[0m 缺少 num_steps，跳过 {fpath}")
                continue

            # 添加方法名与种子
            df["Method"] = human
            df["seed"] = infer_seed_from_path(fpath)

            # 平滑可用的指标列
            for col in ["return", "success_rate", "violation_rate", "average_steps"]:
                if col in df.columns:
                    df[f"{col}_smooth"] = smooth(df[col].to_numpy(), smooth_radius)

            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    return result


def choose_y_column(df: pd.DataFrame, metric: str) -> str:
    """
    根据 METRIC 选择 y 轴列，优先使用平滑列。
    """
    mapping = {
        "return": ["return_smooth", "return"],
        "success_rate": ["success_rate_smooth", "success_rate"],
        "violation_rate": ["violation_rate_smooth", "violation_rate"],
        "avg_steps": ["average_steps_smooth", "average_steps"],
    }
    for cand in mapping.get(metric, []):
        if cand in df.columns:
            return cand
    # 兜底：如果 metric 找不到，尽量用 return / success_rate 之一
    for cand in ["return_smooth", "return", "success_rate_smooth", "success_rate"]:
        if cand in df.columns:
            return cand
    raise ValueError("没有可用的 y 轴列，请检查 log.csv 列名与 METRIC 设置。")


def make_plot(df: pd.DataFrame, out_path: Path, use_ci: bool = True, metric: str = "return") -> None:
    if df.empty:
        print("\033[31m[ERROR]\033[0m DataFrame 为空，无法绘图（没有找到 log.csv 或列缺失）。")
        return

    y_col = choose_y_column(df, metric)

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 7.0))
    ax.set(xlabel="Number of steps", ylabel={
        "return": "Discounted return (per episode mean)",
        "success_rate": "Success rate (per episode mean)",
        "violation_rate": "Violation rate (per episode mean)",
        "avg_steps": "Steps per episode (mean)"
    }.get(metric, y_col))

    # 自动刻度
    if "num_steps" in df.columns:
        max_steps = float(df["num_steps"].max())
        if max_steps > 0:
            ticks = np.linspace(0, max_steps, 8)
            ax.set_xticks(ticks)

    errorbar: Tuple[str, int] | Tuple[str, int] = ("ci", 90) if use_ci else ("sd", 1)

    sns.lineplot(
        data=df,
        x="num_steps",
        y=y_col,
        errorbar=errorbar,
        hue="Method",
        estimator="mean",
        ax=ax,
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)

    # 稀疏化标签
    for lab in ax.xaxis.get_ticklabels()[::2]:
        lab.set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"\033[32m[SAVED]\033[0m 图已保存到：{out_path}")
    plt.show()


def main():
    df = process_eval_results(
        env=ENV,
        experiments=EXPERIMENTS,
        name_mapping=NAME_MAPPING,
        smooth_radius=SMOOTH_RADIUS,
    )
    if df.empty:
        print("\033[31m[ERROR]\033[0m 未拼出任何数据：请检查 ENV/EXPERIMENTS 路径与 log.csv 是否存在。")
        return
    make_plot(df, OUT_PATH, use_ci=USE_CI, metric=METRIC)


if __name__ == "__main__":
    main()








# import os

# import numpy as np
# import pandas as pd
# import seaborn as sns
# from matplotlib import pyplot as plt

# sns.set_theme(font_scale=2.9)


# def main():
#     env = 'PointLtl2-v0'
#     # env = 'LetterEnv-v0'
#     # env = 'FlatWorld-v0'
#     # experiments = ['noactivesampling', 'nocurriculum']
#     # experiments = ['deepltl', 'nocurriculum']
#     experiments = ['zones_paper_s1', 'zones_paper_s1_shapping']
#     # experiments = ['deepset_complex', 'gcrl', 'ltl2action']
#     name_mapping = {'deepltl': 'Curriculum', 'gcrl': 'GCRL-LTL', 'ltl2action': 'LTL2Action', 'deepset': 'DeepLTL', 'nocurriculum': 'No curriculum', 'deepset_complex': 'DeepLTL'}
#     df = process_eval_results(env, experiments, name_mapping)
#     ci = True

#     fig, ax = plt.subplots(1, 1, figsize=(9,7))
#     ax.set(ylabel='Discounted return', yticks=np.arange(0, 1.01, 0.1), xlabel='Number of steps', xticks=np.arange(0, 16, 2) * 1000000)
#     errorbar = ('ci', 90) if ci else ('sd', 1)
#     sns.lineplot(df, x='num_steps', y='return_smooth', errorbar=errorbar, hue='Method', ax=ax)
#     # sns.relplot(df, x='num_steps', y='return', kind='line', ci=ci, hue='seed', col='Method')
#     # plt.savefig(os.path.expanduser('~/work/dphil/iclr-deepltl/figures/training_letter.pdf'))
#     handles, labels = ax.get_legend_handles_labels()
#     ax.legend(handles=handles, labels=labels)  # remove title='Method'

#     for label in ax.xaxis.get_ticklabels()[::2]:
#         label.set_visible(False)

#     # plt.savefig('/home/matier/tmp/curves_ablation.pdf', bbox_inches='tight')
#     plt.savefig('/home/gh/公共/zh/image/curves_ablation.pdf', bbox_inches='tight')
#     plt.show()


# def process_eval_results(env: str, experiments: list[str], name_mapping=None, smooth_radius=10):
#     dfs = []
#     for experiment in experiments:
#         # path = f'eval_results/{env}/{experiment}'
#         path = f'experiments/ppo/{env}/{experiment}'
#         files = [f for f in os.listdir(path) if f.endswith('.csv')]
#         for file in files:
#             # if experiment == 'super_comp' and (file.startswith('1') or file.startswith('3')):
#             #   continue
#             df = pd.read_csv(f'{path}/{file}')
#             name = name_mapping.get(experiment, experiment)
#             df['Method'] = name
#             df['seed'] = int(file.split('.')[0])
#             for col in ['success_rate', 'violation_rate', 'average_steps', 'return']:
#                 df[f'{col}_smooth'] = smooth(df[col], smooth_radius)
#             dfs.append(df)
#         print(f'Loaded {len(files)} files for {name_mapping.get(experiment, experiment)}')
#     result = pd.concat(dfs)
#     if result.isna().any().any():
#         print('Warning: data contains NaN values')
#     return result


# def smooth(row, radius):
#     """
#     Computes the moving average over the given row of data. Returns an array of the same shape as the original row.
#     """
#     y = np.ones(radius)
#     z = np.ones(len(row))
#     return np.convolve(row, y, 'same') / np.convolve(z, y, 'same')


# if __name__ == '__main__':
#     main()


