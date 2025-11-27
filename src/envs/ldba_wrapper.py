# --- 放置于: src/envs/ldba_wrapper.py ---

import functools
from typing import Any, SupportsFloat, Iterable

import gymnasium
from gymnasium.core import WrapperObsType, WrapperActType

from envs import get_env_attr
from ltl.automata import ltl2ldba, LDBA, LDBASequence


class LDBAWrapper(gymnasium.Wrapper):
    """
    跟踪 LTL 目标满足性的包装器：内部维护一个 LDBA，并把其状态加入观测。
    变更点：
    - 优先使用 info['L_stab'] 作为真值集合（来自 BeliefWrapper 的稳健命题）；
      若不存在，则回退到 info['propositions']（单帧布尔命题）。
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, gymnasium.spaces.Dict):
            raise ValueError('LDBA wrapper requires dict observations')
        if 'goal' not in env.observation_space.spaces:
            raise ValueError('LDBA wrapper requires goal in observation space')

        self.terminate_on_acceptance: bool = False
        self.ldba: LDBA | None = None
        self.ldba_state: Any = None
        self.num_accepting_visits: int = 0
        self.obs: WrapperObsType | None = None
        self.info: dict[str, Any] | None = None

    # ---------- 小工具：从 info 中抽取“真命题集合” ----------
    def _extract_true_props(self, info: dict[str, Any]) -> set[str]:
        """
        优先返回稳健命题集合 L_stab；如果没有，则返回布尔命题集合 propositions。
        """
        """优先返回稳健命题集合 L_stab；否则回退到布尔命题集合 propositions。"""
        stable = info.get('L_stab', None)
        if stable is not None:
            info['ldba_truth_source'] = 'L_stab'
            return set(stable)
        info['ldba_truth_source'] = 'propositions'
        if not self._warned_no_l_stab:
            print("[LDBA] L_stab not found in info; fallback to 'propositions'.")
            self._warned_no_l_stab = True
        return set(info.get('propositions', []))


    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # EPSILON 转移特殊处理（用于 LDBA epsilon-closure）
        if (action == LDBASequence.EPSILON).all():
            obs, reward, terminated, truncated, info = self.obs, 0.0, False, False, self.info  # type: ignore
            take_epsilon = True
        else:
            assert not (action == LDBASequence.EPSILON).any()
            obs, reward, terminated, truncated, info = super().step(action)
            take_epsilon = False
            self.obs = obs
            self.info = info

        # 取“真命题集合”（优先 L_stab）
        true_set = self._extract_true_props(info)
        # LDBA 状态推进
        new_ldba_state, accepting = self.ldba.get_next_state(self.ldba_state, true_set, take_epsilon)  # type: ignore
        # active_props = self._truth_from_info(info)
        # new_ldba_state, accepting = self.ldba.get_next_state(self.ldba_state, active_props, take_epsilon)

        if new_ldba_state != self.ldba_state:
            self.ldba_state = new_ldba_state
            info['ldba_state_changed'] = True

        # 完成观测（把 ldba/ldba_state 与用于推进的真命题写回 obs）
        self.complete_observation(obs, info, true_set)

        # 接受状态处理：有限规格收敛则可终止；无限时域则计数 Büchi 访问
        if self.terminate_on_acceptance and accepting:
            terminated = True
            info['success'] = True
        if accepting:
            self.num_accepting_visits += 1

        # 底部强连通分量且非接受：违规（死区）终止
        scc = self.ldba.state_to_scc[self.ldba_state]  # type: ignore
        if scc.bottom and not scc.accepting:
            terminated = True
            info['violation'] = True

        # 记录诊断信息
        info['accepting'] = accepting
        info['num_accepting_visits'] = self.num_accepting_visits
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        self._warned_no_l_stab = False  # 新局重置
        obs, info = super().reset(seed=seed, options=options)
        self.obs = obs
        self.info = info

        # 构造/重置 LDBA
        self.ldba = self.construct_ldba(obs['goal'])
        self.terminate_on_acceptance = self.ldba.is_finite_specification()
        self.ldba_state = self.ldba.initial_state
        self.num_accepting_visits = 0

        # 初始推进一次（基于当前 info 的真命题集合），只更新观测不改变状态
        true_set = self._extract_true_props(info)
        self.complete_observation(obs, info, true_set)
        info['ldba_state_changed'] = True
        return obs, info

    def complete_observation(self, obs: WrapperObsType, info: dict[str, Any], true_set: Iterable[str]):
        """
        将 LDBA 对象、当前 LDBA 状态与“用于推进的真命题集合”写入观测 obs。
        """
        obs['ldba'] = self.ldba
        obs['ldba_state'] = self.ldba_state
        # 将用于推进的命题集回写到 obs['propositions']，便于上层记录/可视化
        obs['propositions'] = set(true_set)

    @functools.cache
    def construct_ldba(self, formula: str) -> LDBA:
        propositions = get_env_attr(self.env, 'get_propositions')()
        ldba = ltl2ldba(formula, propositions, simplify_labels=False)
        possible_assignments = get_env_attr(self.env, 'get_possible_assignments')()
        ldba.prune(possible_assignments)
        ldba.complete_sink_state()
        ldba.compute_sccs()
        initial_scc = ldba.state_to_scc[ldba.initial_state]
        if initial_scc.bottom and not initial_scc.accepting:
            raise ValueError(f'The language of the LDBA for {formula} is empty.')
        return ldba
