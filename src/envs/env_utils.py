from typing import Callable, Optional

import gymnasium
from gymnasium.wrappers import FlattenObservation, TimeLimit
from gymnasium import spaces

from envs.remove_trunc_wrapper import RemoveTruncWrapper


# --------- 通用工具：向下取属性 ---------
def get_env_attr(env, attr: str):
    """获取底层 env 的属性（透过 wrapper 链）"""
    if hasattr(env, attr):
        return getattr(env, attr)
    if hasattr(env, 'env'):
        return getattr(env.unwrapped, attr)
    else:
        raise AttributeError(f'Attribute {attr} not found in env.')
def _debug_wrapper_chain(env):
    i, e = 0, env
    names = []
    while hasattr(e, "env"):
        names.append(type(e).__name__)
        e = e.env
        i += 1
    names.append(type(e).__name__)  # base
    print("[WRAPPERS]", " -> ".join(reversed(names)))


# --------- 入口：构造环境并按训练/评测装配 wrapper ---------
def make_env(
        name: str,
        sampler: Callable[[list[str]], Callable],
        max_steps: Optional[int] = None,
        render_mode: str | None = None,
        sequence: bool = False,
        # ==== 评测期命题噪声（默认关闭）====
        eval_noise_enable: bool = False,
        noise_p_miss: float = 0.0,
        noise_p_false: float = 0.0,
        noise_seed: int | None = 123,
        # ==== 命题信念化（Belief）====
        belief_enable: bool = False,
        belief_theta_up: float = 0.8,
        belief_theta_down: float = 0.2,
        belief_k_stab: int = 3,
        belief_ema_lambda: float = 0.8,
        belief_k: float = 5.0,
        belief_m: float = 0.0,
):
    """
    装配顺序（关键）：
      SafetyGymWrapper → FlattenObservation  → [BeliefWrapper] →
        (训练) SequenceWrapper
        (评测) LTLWrapper → [PropositionNoiseWrapper] → LDBAWrapper
      → TimeLimit → RemoveTruncWrapper
    """
    from envs.seq_wrapper import SequenceWrapper
    from envs.ldba_wrapper import LDBAWrapper
    from envs.ltl_wrapper import LTLWrapper

    # 我们的附加封装
    # - 命题信念化
    try:
        from envs.belief_wrapper import BeliefWrapper, BeliefParams
        _has_belief = True
    except Exception:
        BeliefWrapper, BeliefParams, _has_belief = None, None, False

    # - 命题噪声
    try:
        from envs.proposition_noise_wrapper import PropositionNoiseWrapper
        _has_noise = True
    except Exception:
        PropositionNoiseWrapper, _has_noise = None, False

    # ---- 基础环境 ----
    if is_safety_gym_env(name):
      # SafetyGym：返回 Dict(...)，随后 Flatten 为 Box
        env = make_safety_gym_env(name, render_mode)
        max_steps = max_steps or 1000
    elif name.startswith('Letter'):
        env = make_letter_env(name, render_mode)
        max_steps = max_steps or 75
    elif name.startswith('FlatWorld'):
        env = make_flatworld_env(name)
        max_steps = max_steps or 500
    else:
        raise ValueError(f'Unknown environment: {name}')

    # ---- （可选）命题信念化：保证不改 obs 结构，只写 info ----
    if belief_enable:
        if not _has_belief:
            raise ImportError("BeliefWrapper 未找到，请确认文件 envs/belief_wrapper.py 存在且类名一致。")
        params = BeliefParams(
            k=belief_k, m=belief_m,
            ema_lambda=belief_ema_lambda,
            theta_up=belief_theta_up,
            theta_down=belief_theta_down,
            k_stab=belief_k_stab
        )
        env = BeliefWrapper(env, params=params)

    # ---- 采样任务（LTL 序列 / 公式）----
    propositions = get_env_attr(env, 'get_propositions')()
    sample_task = sampler(propositions)

    # ---- 训练 / 评测 分支 ----
    if sequence:
        # === 训练：用 reach-avoid 序列，且不加噪声（C→C）===
        env = SequenceWrapper(env, sample_task)
    else:
        # === 评测：用 LTL 公式 + （可选）命题噪声 + LDBA ===
        env = LTLWrapper(env, sample_task)

        # （可选）对 info['propositions'] 加噪声；若已信念化，优先在 L_stab 上扰动
        if eval_noise_enable and (noise_p_miss > 0.0 or noise_p_false > 0.0):
            if not _has_noise:
                raise ImportError("PropositionNoiseWrapper 未找到，请确认文件 envs/proposition_noise_wrapper.py 存在。")
            # 颜色命题互斥组（同一时刻最多一个为真）
            mutex_groups = [set(propositions)]
            env = PropositionNoiseWrapper(
                env,
                ap_list=propositions,
                p_miss=noise_p_miss,
                p_false=noise_p_false,
                seed=noise_seed,
                mutex_groups=mutex_groups
            )

        env = LDBAWrapper(env)

    # ---- 限制步数 + 去除 trunc 标记 ----
    env = TimeLimit(env, max_episode_steps=max_steps)
    env = RemoveTruncWrapper(env)

    # ---- 形状兜底：此时应为 Dict({'features': Box(...), 'goal': ..., ...}) ----
    if not isinstance(env.observation_space, spaces.Dict) or 'features' not in env.observation_space.spaces:
        raise RuntimeError(
            f"[env_utils] 观测空间不符合预期：{env.observation_space}. "
            f"期望为 Dict({'features': Box(...)} + 其它键)。"
        )
    _debug_wrapper_chain(env)
    return env


# --------- 环境族别与子构造 ---------
def is_safety_gym_env(name: str) -> bool:
    """判断是否为 Safety-Gym 家族环境。"""
    return any([name.startswith(agent_name) for agent_name in ['Point', 'Car', 'Racecar', 'Doggo', 'Ant']])


def make_safety_gym_env(name: str, render_mode: str | None = None):
    """
    Safety-Gym：
      SafetyGymWrapper 输出 Dict(...) → FlattenObservation 拉平成 Box(...)
      这样 Sequence/LTL 的 'features' = Box 即可被模型读取
    """
    # noinspection PyUnresolvedReferences
    import safety_gymnasium
    from envs.zones.safety_gym_wrapper import SafetyGymWrapper

    env = safety_gymnasium.make(name, render_mode=render_mode)
    env = SafetyGymWrapper(env)
    env = FlattenObservation(env)     # 关键：将 Dict(obs) 展平为 Box
    return env


def make_letter_env(name: str, render_mode: str | None = None):
    import envs.letter_world
    env = gymnasium.make(name, render_mode=render_mode)
    return env


def make_flatworld_env(name: str):
    import envs.flatworld
    env = gymnasium.make(name)
    return env
