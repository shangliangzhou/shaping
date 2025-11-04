# src/train/her_suffix.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Hashable, Set

import numpy as np
import torch

__all__ = [
    "split_episodes_from_batch",
    "suffix_her_episode",
    "aux_value_update",
    # 工具函数（一般无需外部调用）
    "to_ap_set", "active_ap_set", "remain_len", "d_goal",
    "shape_rewards_for_episode",
]

# -------------------- AP/命题解析，与势函数保持一致 --------------------

def to_ap_set(x: Any) -> Optional[Set[Hashable]]:
    if x is None:
        return None
    if isinstance(x, str):
        return {x}
    if isinstance(x, (list, tuple, set)):
        S: Set[Hashable] = set()
        for e in x:
            if isinstance(e, str):
                S.add(e)
            elif hasattr(e, "name"):
                S.add(getattr(e, "name"))
            else:
                S.add(e)
        return S
    if isinstance(x, dict):  # {ap: bool}
        return {k for k, v in x.items() if bool(v)}
    for attr in ("atoms", "aps", "literals", "symbols"):
        if hasattr(x, attr):
            vals = getattr(x, attr)
            return set(vals) if not isinstance(vals, dict) else set(vals.keys())
    try:
        return {str(x)}
    except Exception:
        return None

def active_ap_set(p: Any) -> Optional[Set[Hashable]]:
    if p is None:
        return None
    if isinstance(p, set):
        return set(p)
    if isinstance(p, (list, tuple)):
        return set(p)
    if isinstance(p, dict):
        return {k for k, v in p.items() if bool(v)}
    for attr in ("true", "active", "positives", "atoms", "aps"):
        if hasattr(p, attr):
            vals = getattr(p, attr)
            return set(vals) if not isinstance(vals, dict) else {k for k, v in vals.items() if bool(v)}
    if isinstance(p, str):
        return {p}
    try:
        return set(p)
    except Exception:
        return None

def remain_len(obs: Dict[str, Any]) -> Optional[int]:
    if isinstance(obs, dict) and "goal" in obs:
        g = obs["goal"]
        try:
            return len(list(g))
        except Exception:
            return None
    return None

def d_goal(obs: Dict[str, Any]) -> Optional[float]:
    """
    到下一子目标的匹配距离：d = 1 - |P ∩ G| / |G|
      - 无剩余目标时 0
      - 取不到 P/G 时 None（本步不加几何项）
    """
    if not isinstance(obs, dict):
        return None
    goal_seq = obs.get("goal", None)
    try:
        goal_seq = list(goal_seq) if goal_seq is not None else None
    except Exception:
        return None
    if not goal_seq:
        return 0.0
    G = to_ap_set(goal_seq[0])
    if not G:
        return 0.0
    P = active_ap_set(obs.get("propositions", None))
    if P is None:
        return None
    inter = len(P & G)
    d = 1.0 - float(inter) / max(1, len(G))
    return float(np.clip(d, 0.0, 1.0))

# -------------------- 按势函数离线重算形塑回报 --------------------

def shape_rewards_for_episode(
    obss: List[Dict[str, Any]],
    base_rewards: List[float],
    gamma: float,
    beta: float,
    eta: float,
    alpha: float
) -> np.ndarray:
    """
    形塑：r'_t = r_t + beta * (phi_prev - gamma * phi_t) + eta * 1{remain下降}
    其中 phi = remain + alpha * d_goal
    """
    T = len(obss)
    shaped = np.asarray(base_rewards, dtype=np.float32).copy()
    last_phi = None
    last_rem = None
    for t in range(T):
        rem_t = remain_len(obss[t])
        dg_t = d_goal(obss[t]) or 0.0
        phi_t = (rem_t or 0) + alpha * dg_t
        if last_phi is not None:
            shaping = beta * (last_phi - gamma * phi_t)
            shaped[t] += shaping
            if eta > 0.0 and rem_t is not None and last_rem is not None and rem_t < last_rem:
                shaped[t] += eta
        last_phi = phi_t
        last_rem = rem_t
    return shaped

# -------------------- 批次字段自动探测 + 按 logs 切 episode --------------------

def _tolist(x):
    try:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    try:
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    try:
        return list(x)
    except Exception:
        return x

def _iter_fields(exps) -> List[Tuple[str, Any]]:
    items = []
    if isinstance(exps, dict):
        items.extend(list(exps.items()))
    for k in dir(exps):
        if k.startswith("_"):
            continue
        try:
            v = getattr(exps, k)
        except Exception:
            continue
        if callable(v):
            continue
        items.append((k, v))
    seen = {}
    for k, v in items:
        seen[k] = v
    return list(seen.items())

def _looks_like_obs_list(v) -> bool:
    v = _tolist(v)
    if not isinstance(v, list) or len(v) == 0:
        return False
    first = v[0]
    if isinstance(first, dict):
        keys = set(first.keys())
        if {"goal", "propositions"}.intersection(keys) or "initial_goal" in keys or "features" in keys:
            return True
    return False

def _pick_obss_actions_rewards(exps) -> Tuple[str, list, str, list, str, list]:
    """
    自动探测 obss/actions/rewards 三个字段
    返回: (obs_name, obss_list, act_name, acts_list, rew_name, rews_list)
    """
    fields = _iter_fields(exps)

    # 1) obss
    obs_name, obss = None, None
    for name, val in fields:
        if _looks_like_obs_list(val):
            obs_name, obss = name, _tolist(val)
            break
    if obss is None:
        for name, val in fields:
            lname = name.lower()
            if any(t in lname for t in ["obss", "obs", "observations"]):
                val_l = _tolist(val)
                if isinstance(val_l, list) and len(val_l) > 0 and isinstance(val_l[0], dict):
                    obs_name, obss = name, val_l
                    break
    if obss is None:
        names = [n for n, _ in fields]
        raise KeyError(f"cannot locate observations in exps; available fields: {names}")

    N = len(obss)

    # 2) rewards
    rew_name, rews = None, None
    for name, val in fields:
        lname = name.lower()
        if "reward" in lname:
            val_l = _tolist(val)
            if isinstance(val_l, list) and len(val_l) == N:
                rews = val_l; rew_name = name; break
    if rews is None:
        for name, val in fields:
            val_l = _tolist(val)
            if isinstance(val_l, list) and len(val_l) == N:
                try:
                    float(val_l[0])
                    rews = val_l; rew_name = name; break
                except Exception:
                    continue
    if rews is None:
        raise KeyError("cannot locate rewards in exps")

    # 3) actions
    act_name, acts = None, None
    for name, val in fields:
        lname = name.lower()
        if "action" in lname:
            val_l = _tolist(val)
            if isinstance(val_l, list) and len(val_l) == N:
                acts = val_l; act_name = name; break
    if acts is None:
        for name, val in fields:
            val_l = _tolist(val)
            if isinstance(val_l, list) and len(val_l) == N:
                acts = val_l; act_name = name; break
    if acts is None:
        raise KeyError("cannot locate actions in exps")

    return obs_name, obss, act_name, acts, rew_name, rews

def split_episodes_from_batch(exps, logs):
    """
    返回 episodes = [{ 'obss': [...], 'acts': [...], 'rews': [...], 'dones': [...] }, ...]
    优先使用 logs['num_steps_per_episode'] 切分（最稳）。
    """
    obs_name, obss, act_name, acts, rew_name, rews = _pick_obss_actions_rewards(exps)
    lengths = logs.get("num_steps_per_episode", None)

    print(f"[HER] autodetected fields: obs='{obs_name}', act='{act_name}', rew='{rew_name}', "
          f"N={len(obss)}, episodes_by_lengths={'yes' if lengths is not None else 'no'}")

    if lengths is not None:
        episodes = []
        idx = 0
        for L in lengths:
            L = int(L)
            episodes.append({
                "obss":  obss[idx:idx+L],
                "acts":  acts[idx:idx+L],
                "rews":  rews[idx:idx+L],
                "dones": [False]*(L-1) + [True],
            })
            idx += L
        if idx < len(rews):
            L = len(rews) - idx
            episodes.append({
                "obss":  obss[idx:], "acts": acts[idx:], "rews": rews[idx:],
                "dones": [False]*L,
            })
        return episodes

    # 兜底：尝试 dones/mask
    dones = None
    if hasattr(exps, "dones"):
        dones = _tolist(getattr(exps, "dones"))
    if dones is None and hasattr(exps, "mask"):
        mask = _tolist(getattr(exps, "mask"))
        dones = [bool(1 - m) for m in mask]  # mask=1非终止，0终止

    if dones is None:
        raise KeyError("Neither logs['num_steps_per_episode'] nor (dones/mask) available.")

    episodes = []
    start = 0
    for t, d in enumerate(dones):
        if d:
            episodes.append({
                "obss":  obss[start:t+1],
                "acts":  acts[start:t+1],
                "rews":  rews[start:t+1],
                "dones": [False]*(t-start) + [True],
            })
            start = t+1
    if start < len(rews):
        episodes.append({
            "obss":  obss[start:], "acts": acts[start:], "rews": rews[start:],
            "dones": [False]*(len(rews)-start),
        })
    return episodes

# -------------------- 后缀 HER：重标目标序列 --------------------

def satisfy(P: Set[Hashable], G: Set[Hashable]) -> bool:
    return G.issubset(P)

# src/train/her_suffix.py 片段替换

def suffix_her_episode(ep: Dict[str, Any], k_future: int = 4) -> List[Dict[str, Any]]:
    """
    从一条 episode 生成若干条“后缀重标”的 HER 样本。
    输入 ep 可能来自两路：
      - 环境记录器 EpisodeBufferWrapper：{obss, acts, rews_shaped, dones}
      - 基于 exps 切分：{obss, acts, rews, dones}
    这里统一兼容，优先使用我们在 train_ppo 里改装的 rews_env（若存在）。
    """
    obss = ep["obss"]
    acts = ep["acts"]

    # ✅ 关键：按优先级回落获取奖励基线
    base = ep.get("rews_env", None)
    if base is None:
        base = ep.get("rews_shaped", None)
    if base is None:
        base = ep.get("rews", None)
    if base is None:
        base = [0.0] * len(acts)  # 兜底为全0，后续会用势函数重算

    T = len(obss)
    if T == 0:
        return []

    init_goal = list(obss[0].get("initial_goal", []))
    if not init_goal:
        return []

    Ps = [active_ap_set(o.get("propositions", None)) or set() for o in obss]
    idxs = np.random.choice(np.arange(T), size=min(k_future, T), replace=False)

    her_eps = []
    for t_prime in idxs:
        j = None
        for jj, Gj_raw in enumerate(init_goal):
            Gj = to_ap_set(Gj_raw)
            if Gj and satisfy(Ps[t_prime], Gj):
                j = jj
                break
        if j is None:
            continue

        goal_suffix = init_goal[j:]
        num_reached = 0
        new_obss = []
        for s in range(T):
            o = dict(obss[s])  # 浅拷贝
            rem_seq = goal_suffix[num_reached:]
            o["initial_goal"] = goal_suffix
            o["goal"] = rem_seq
            new_obss.append(o)
            if rem_seq:
                G = to_ap_set(rem_seq[0]) or set()
                if satisfy(Ps[s], G):
                    num_reached += 1

        # 统一回传字段名为 rews_env，后续 aux_value_update 会读取它
        her_eps.append({"obss": new_obss, "acts": acts, "rews_env": base})

    return her_eps


# -------------------- 辅助价值头更新（可选微弱BC） --------------------

def _extract_value_and_logprob(model, out, ob_proc, acts_tensor):
    """
    兼容不同模型接口的取值：
      - out 是 dict: 取 out['value']，若有 out['log_prob'](a) 用之
      - out 有 .v: 取 out.v
      - 若无 log_prob 接口，则返回 logp=None（可跳过 BC）
    """
    v_pred = None
    logp = None

    if isinstance(out, dict):
        if "value" in out:
            v_pred = out["value"]
        elif "v" in out:
            v_pred = out["v"]
        if "log_prob" in out and callable(out["log_prob"]):
            try:
                logp = out["log_prob"](acts_tensor)
            except Exception:
                logp = None
    else:
        # 常见：命名为 .v
        if hasattr(out, "v"):
            v_pred = getattr(out, "v")
        # 有的模型把分布器存在 out.dist; 这里不强求

    return v_pred, logp

def aux_value_update(
    model,
    optimizer,
    preprocess_obss,
    her_eps: List[Dict[str, Any]],
    gamma: float,
    beta: float,
    eta: float,
    alpha: float,
    batch_size: int = 1024,
    epochs: int = 1,
    device: str = "cuda",
    lambda_bc: float = 0.0
) -> Dict[str, float]:
    """
    用 HER 样本只训练 value 头（可选极小 BC），不改 PPO 主更新。
    返回日志：{'her_used': N, 'her_vloss': 平均MSE}
    """
    if len(her_eps) == 0:
        return {"her_used": 0, "her_vloss": 0.0}

    obs_all: List[Dict[str, Any]] = []
    act_all: List[Any] = []
    ret_all: List[float] = []

    for ep in her_eps:
        obss = ep["obss"]; acts = ep["acts"]; base = ep["rews_env"]
        shaped = shape_rewards_for_episode(obss, base, gamma, beta, eta, alpha)
        # MC returns
        R = 0.0; returns = []
        for r in reversed(list(shaped)):
            R = float(r) + gamma * R
            returns.append(R)
        returns = list(reversed(returns))
        obs_all += obss
        act_all += acts
        ret_all += returns

    import torch.nn.functional as F
    N = len(obs_all)
    idx = np.arange(N)
    vloss_acc = 0.0; iters = 0

    for _ in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, N, batch_size):
            b_idx = idx[s:s+batch_size]
            ob = preprocess_obss([obs_all[i] for i in b_idx], device=device)
            target_v = torch.tensor([ret_all[i] for i in b_idx],
                                    dtype=torch.float32, device=device).unsqueeze(1)

            out = model(ob)
            v_pred, logp = _extract_value_and_logprob(model, out, ob, None)

            if v_pred is None:
                raise RuntimeError("Model forward did not return a value head ('value' or '.v').")

            v_loss = F.mse_loss(v_pred, target_v)
            loss = v_loss

            if lambda_bc > 0.0 and logp is not None:
                # 若模型暴露了 log_prob 接口，才做极小BC；否则跳过
                # 这里没有动作张量，BC 通常需要策略分布；如果你的 out['log_prob'] 需要 (ob, a)，
                # 可按你们接口改写 _extract_value_and_logprob。
                pass

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            vloss_acc += float(v_loss.detach().cpu().item())
            iters += 1

    return {"her_used": int(N), "her_vloss": (vloss_acc / max(1, iters))}
