# -*- coding: utf-8 -*-
"""
zh_eval_plot.py (robust version)
读取各方法的 log.csv，计算：
  1) success-rate 的归一化 AUC
  2) 达到 success >= 0.9 的样本步数
并绘制两张柱状图（均值±标准差），输出 summary_metrics.csv。

用法:
    python src/evaluation/zh_eval_plot.py
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== 1) 在这里配置你的日志文件 ======
# 每个方法可以是 单个 str 路径 或 List[str]（多 seed）
METHODS = {
    "Zones S1": [
        "experiments/ppo/PointLtl2-v0/zones_paper_s1/1/log.csv"
    ],
    "Zones S1 + Shaping": [
        # 注意你之前拼写是 shapping（双 p），这里沿用
        "experiments/ppo/PointLtl2-v0/zones_paper_s1_shapping/1/log.csv"
    ],
    "PPO Shaping + alpha=0.5": [
        "experiments/ppo/PointLtl2-v0/ppo_shaping_alpha05/1/log.csv"
    ],
    "PPO Shaping + HER": [
        "experiments/ppo/PointLtl2-v0/ppo_shaping_her/1/log.csv"
    ],
}

DEBUG_PRINT = True  # 打开可看到每个 CSV 选中的列

STEP_CANDIDATES = ["num_steps", "steps", "total_steps", "t", "global_steps"]
SUCCESS_CANDIDATES = [
    # 常见命名
    "success_rate", "avg_goal_success", "goal_success", "goal_success_rate",
    "success", "success_mean",
    # 日志中的 P（成功率）系列（含希腊 μ/µ）
    "pμ", "pµ", "p_mu", "p mean", "pmean", "p_mean", "pmu", "p mu",
    "pμ_mean", "pµ_mean"
]

# ====== 2) 工具函数 ======
def _coerce_numeric_series(s: pd.Series, is_step: bool = False) -> pd.Series:
    """
    把字符串安全转为数值：
      - 支持 '0.91±0.03'、'91.3%'、'1,500,000'、'15_000_000'、科学计数法等
      - 成功率>1 视为百分比自动 /100 归一化
    """
    ss = s.astype(str)
    ss = ss.str.replace(",", "", regex=False).str.replace("_", "", regex=False).str.strip()

    # 百分号：'91.3%' -> 0.913
    has_percent = ss.str.contains("%")
    if has_percent.any():
        num = ss.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")[0]
        vals = pd.to_numeric(num, errors="coerce")
        vals = vals / 100.0
        return vals

    # 一般：提取第一个数字
    num = ss.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")[0]
    vals = pd.to_numeric(num, errors="coerce")

    if is_step:
        return vals

    # success 若明显像 0~100，自动 /100
    try:
        vmax = float(vals.max(skipna=True))
        if vmax > 1.5:
            vals = vals / 100.0
    except Exception:
        pass
    return vals


def _pick_step_col(df: pd.DataFrame) -> str | None:
    cols = df.columns
    for c in STEP_CANDIDATES:
        if c in cols:
            return c
    # 兜底：包含 'step'
    for c in cols:
        if "step" in c:
            return c
    return None


def _pick_success_col(df: pd.DataFrame) -> str | None:
    """
    更严格地挑选成功率列：
    1) 先走 SUCCESS_CANDIDATES 精确匹配
    2) 再走正则捕捉 'Pμ/µ/P_mu/P mean' 等形态
    3) 最后在候选里选择“值在 [0,1] 且方差>0”的那一列
    """
    cols = list(df.columns)

    # 1) 直接候选
    for c in SUCCESS_CANDIDATES:
        if c in cols:
            return c

    # 2) 正则：P + (μ/µ/_?mu) + (可选 mean/下划线)
    mu_block = r"(μ|µ|_?mu|\s*mu)"
    patterns = [
        rf"^p{mu_block}(\b|_|$)",              # pμ / p_mu / p mu / pµ
        rf"^p\s*{mu_block}\s*(_?mean|\b|$)",   # pμ_mean / p mu mean
        r"^(p[_\s]*mean|pmean|p_mean)\b",      # pmean / p_mean
        r"success"                              # 包含 success 字样
    ]
    pool = [c for c in cols if any(re.search(p, c) for p in patterns)]

    # 排除明显不对的列名
    pool = [c for c in pool if not any(bad in c for bad in ["policy", "value", "loss"])]

    # 3) 在 pool 里选最像成功率的列（均值在 [0,1]，方差>0 优先）
    best_col, best_score = None, -1.0
    for c in pool:
        vals = _coerce_numeric_series(df[c], is_step=False)
        m = np.nanmean(vals)
        v = np.nanstd(vals)
        if np.isnan(m): 
            continue
        score = 0.0
        if 0.0 <= m <= 1.0:
            score += 1.0
        score += 0.1 * float(v)  # 有波动更像“曲线”，略加分
        if score > best_score:
            best_score = score
            best_col = c
    return best_col


def read_one_csv(path: str) -> pd.DataFrame:
    """读取单个 log.csv，返回两列：steps, success（成功率∈[0,1]）"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)

    # 列名统一为小写、去空格
    rename = {c: str(c).strip().lower() for c in df.columns}
    df.rename(columns=rename, inplace=True)

    step_col = _pick_step_col(df)
    succ_col = _pick_success_col(df)

    if step_col is None or succ_col is None:
        raise KeyError(
            f"Cannot locate steps/success columns in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    if DEBUG_PRINT:
        print(f"[OK] {os.path.basename(path)} -> steps='{step_col}', success='{succ_col}'")

    out = df[[step_col, succ_col]].copy()
    out.columns = ["steps", "success"]

    out["steps"] = _coerce_numeric_series(out["steps"], is_step=True)
    out["success"] = _coerce_numeric_series(out["success"], is_step=False)

    out = out.dropna(subset=["steps", "success"]).sort_values("steps")
    out = out.groupby("steps", as_index=False).tail(1)
    out["success"] = out["success"].clip(0.0, 1.0)
    return out


def auc_normalized(df: pd.DataFrame, max_steps: float | None = None) -> float:
    """success vs steps 的归一化面积（除以区间长度），越大越好"""
    if df.empty:
        return 0.0
    x = df["steps"].to_numpy(dtype=float)
    y = df["success"].to_numpy(dtype=float)
    if max_steps is None:
        max_steps = float(np.max(x))
    if x[0] > 0.0:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, y[0])
    if x[-1] < max_steps:
        x = np.append(x, max_steps)
        y = np.append(y, y[-1])
    area = np.trapezoid(y, x)  # 替代 trapz
    return float(area / max_steps) if max_steps > 0 else 0.0


def steps_to_threshold(df: pd.DataFrame, thr: float = 0.9, max_steps: float | None = None) -> float:
    """首次达到 success≥thr 的样本步数；未达则返回 max_steps"""
    if df.empty:
        return float("nan") if max_steps is None else float(max_steps)
    x = df["steps"].to_numpy(dtype=float)
    y = df["success"].to_numpy(dtype=float)
    idx = np.where(y >= thr)[0]
    if len(idx) > 0:
        return float(x[idx[0]])
    if max_steps is None:
        return float(x[-1])
    return float(max_steps)


def bootstrap_pvalue(a: np.ndarray, b: np.ndarray, iters: int = 5000, seed: int = 0) -> float:
    """双样本 bootstrap（检验 mean(a) 与 mean(b) 的差异），返回双尾 p 值；样本不足返回 NaN"""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    obs = np.mean(a) - np.mean(b)
    pool = np.concatenate([a, b], axis=0)
    cnt = 0
    for _ in range(iters):
        ia = rng.choice(len(pool), size=len(a), replace=True)
        ib = rng.choice(len(pool), size=len(b), replace=True)
        diff = np.mean(pool[ia]) - np.mean(pool[ib])
        if abs(diff) >= abs(obs):
            cnt += 1
    return (cnt + 1) / (iters + 1)


def aggregate_one_method(files: list[str]) -> dict:
    runs, max_steps = [], 0.0
    for f in files:
        if not os.path.isfile(f):
            print(f"[WARN] log not found, skip: {f}")
            continue
        try:
            df = read_one_csv(f)
        except Exception as e:
            print(f"[WARN] failed to read {f}: {e}")
            continue
        runs.append(df)
        if not df.empty:
            max_steps = max(max_steps, float(df["steps"].max()))

    if len(runs) == 0:
        return {"runs": [], "max_steps": 0.0, "auc": np.array([]), "s90": np.array([])}

    aucs = [auc_normalized(df, max_steps=max_steps) for df in runs]
    s90  = [steps_to_threshold(df, thr=0.9, max_steps=max_steps) for df in runs]
    return {
        "runs": runs,
        "max_steps": max_steps,
        "auc": np.array(aucs, dtype=float),
        "s90": np.array(s90, dtype=float),
    }

# ====== 3) 主流程 ======
def main():
    results = {}
    for method, files in METHODS.items():
        files = files if isinstance(files, list) else [files]
        results[method] = aggregate_one_method(files)

    # 汇总表
    rows = []
    for m, r in results.items():
        if r["auc"].size == 0:
            rows.append({
                "method": m,
                "auc_norm_mean": float("nan"),
                "auc_norm_std": float("nan"),
                "steps_to_0.9_mean": float("nan"),
                "steps_to_0.9_std": float("nan"),
                "reached_ratio": float("nan"),
            })
            continue
        auc_mean, auc_std = np.mean(r["auc"]), np.std(r["auc"])
        s90_mean, s90_std = np.mean(r["s90"]), np.std(r["s90"])
        reached = np.mean(r["s90"] < r["max_steps"] * 0.999)
        rows.append({
            "method": m,
            "auc_norm_mean": float(auc_mean),
            "auc_norm_std": float(auc_std),
            "steps_to_0.9_mean": float(s90_mean),
            "steps_to_0.9_std": float(s90_std),
            "reached_ratio": float(reached)
        })
    summary = pd.DataFrame(rows).sort_values("method")
    summary.to_csv("summary_metrics.csv", index=False)
    print("Saved summary -> summary_metrics.csv")
    print(summary)

    # 仅对有数据的方法绘图
    methods_order = [m for m in METHODS.keys() if results[m]["auc"].size > 0 and not np.all(np.isnan(results[m]["auc"]))]

    if len(methods_order) == 0:
        print("[ERROR] no valid methods with data; nothing to plot.")
        return

    baseline = methods_order[0]
    for metric in ["auc", "s90"]:
        base_vals = results[baseline][metric]
        for m in methods_order[1:]:
            p = bootstrap_pvalue(base_vals, results[m][metric])
            print(f"[p-value] baseline({baseline}) vs {m} on {metric}: {p:.4f}")

    # --- AUC 柱状图 ---
    auc_means = [float(np.nanmean(results[m]["auc"])) for m in methods_order]
    auc_stds  = [float(np.nanstd(results[m]["auc"]))  for m in methods_order]
    plt.figure()
    x = np.arange(len(methods_order))
    plt.bar(x, auc_means, yerr=auc_stds, capsize=5)
    plt.xticks(x, methods_order, rotation=15, ha="right")
    plt.ylabel("Normalized AUC of success rate")
    plt.title("Higher is better")
    plt.tight_layout()
    plt.savefig("bar_auc.png", dpi=180)
    print("Saved figure -> bar_auc.png")

    # --- Steps-to-0.9 柱状图 ---
    s90_means = [float(np.nanmean(results[m]["s90"])) for m in methods_order]
    s90_stds  = [float(np.nanstd(results[m]["s90"]))  for m in methods_order]
    plt.figure()
    x = np.arange(len(methods_order))
    plt.bar(x, s90_means, yerr=s90_stds, capsize=5)
    plt.xticks(x, methods_order, rotation=15, ha="right")
    plt.ylabel("Steps to reach success ≥ 0.9")
    plt.title("Lower is better")
    plt.tight_layout()
    plt.savefig("bar_steps_to_0.9.png", dpi=180)
    print("Saved figure -> bar_steps_to_0.9.png")


if __name__ == "__main__":
    main()
