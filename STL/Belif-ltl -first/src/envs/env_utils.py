from typing import Callable, Optional

import gymnasium
from gymnasium.wrappers import FlattenObservation, TimeLimit

from envs.remove_trunc_wrapper import RemoveTruncWrapper


def get_env_attr(env, attr: str):
    if hasattr(env, attr):
        return getattr(env, attr)
    if hasattr(env, 'env'):
        return getattr(env.unwrapped, attr)
    else:
        raise AttributeError(f'Attribute {attr} not found in env.')


def make_env(
        name: str,
        sampler: Callable[[list[str]], Callable],
        max_steps: Optional[int] = None,
        render_mode: str | None = None,
        sequence=False,
        # -------- 新增评测噪声参数（默认关闭） --------
        eval_noise_enable: bool = False,
        noise_p_miss: float = 0.05,
        noise_p_false: float = 0.03,
        # noise_sigma_jitter: float = 0.0,
        noise_seed: int | None = 123,
):
    from envs.seq_wrapper import SequenceWrapper
    from envs.ldba_wrapper import LDBAWrapper
    from envs.ltl_wrapper import LTLWrapper
    from envs.proposition_noise_wrapper import PropositionNoiseWrapper  # <--- 新增

    if is_safety_gym_env(name):
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
    

    propositions = get_env_attr(env, 'get_propositions')()
    sample_task = sampler(propositions)
    if not sequence:
        # env = PartiallyOrderedWrapper(env, sample_task)
        env = LTLWrapper(env, sample_task)
        # 2) 再噪声：翻转 LTL 刚写入的 info['propositions']

        # 3) (可选) 评测端加噪声：只在 C→N 时打开
        # 2) 评测端噪声（只在 C→N 时开启）
        if eval_noise_enable and (noise_p_miss > 0.0 or noise_p_false > 0.0):
            mutex_groups = [ {'blue','green','yellow','magenta'} ]
            env = PropositionNoiseWrapper(
                env,
                ap_list=getattr(env, "propositions", None),  # 若 LTLWrapper 暴露全集
                p_miss=noise_p_miss,
                p_false=noise_p_false,
                seed=noise_seed,
                mutex_groups=mutex_groups
            )
        env = LDBAWrapper(env)

        
    else:
        env = SequenceWrapper(env, sample_task)
    env = TimeLimit(env, max_episode_steps=max_steps)
    env = RemoveTruncWrapper(env)
    return env


def is_safety_gym_env(name: str) -> bool:
    return any([name.startswith(agent_name) for agent_name in ['Point', 'Car', 'Racecar', 'Doggo', 'Ant']])


def make_safety_gym_env(name: str, render_mode: str | None = None):
    # noinspection PyUnresolvedReferences
    import safety_gymnasium
    from envs.zones.safety_gym_wrapper import SafetyGymWrapper

    env = safety_gymnasium.make(name, render_mode=render_mode)
    env = SafetyGymWrapper(env)
    env = FlattenObservation(env)
    return env


def make_letter_env(name: str, render_mode: str | None = None):
    import envs.letter_world

    env = gymnasium.make(name, render_mode=render_mode)
    return env


def make_flatworld_env(name: str):
    import envs.flatworld

    env = gymnasium.make(name)
    return env
