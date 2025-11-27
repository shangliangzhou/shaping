from typing import Any, SupportsFloat, Callable

import gymnasium
from gymnasium import spaces
from gymnasium.core import WrapperObsType, WrapperActType

from ltl.automata import LDBASequence
from ltl.logic import Assignment


class SequenceWrapper(gymnasium.Wrapper):
    """
    训练端：为策略加入 reach-avoid 序列（goal_seq），并基于命题标签推进子目标。
    关键改动：
      - 优先使用稳健命题集合 info['L_stab']（来自 Brief/BeliefWrapper）；
        若无则回退到单帧布尔标签 info['propositions']。
      - 在 info 中标注 ldba_truth_source ∈ {'L_stab','propositions'}；
      - 在观测中回填 'propositions_used'（本步用于判定的命题集合，list）。
    其他行为（ε 转移、partial_reward）保持原样。
    """

    def __init__(
        self,
        env: gymnasium.Env,
        sample_sequence: Callable[[], LDBASequence],
        partial_reward: bool = False
        
    ):
        super().__init__(env)
        # 训练端网络通常只吃 'features'，其余键由上层预处理器读取，不做空间声明
        self.observation_space = spaces.Dict({
            'features': env.observation_space,
        })
        self.sample_sequence = sample_sequence
        self.goal_seq = None
        self.num_reached = 0
        self.propositions = set(env.get_propositions())
        self.partial_reward = partial_reward
        self.obs: WrapperObsType | None = None
        self.info: dict[str, Any] | None = None

        # 日志辅助：本局是否已提示过“缺少 L_stab”
        self._warned_no_l_stab: bool = False
        # 为了在 complete_observation 中回填“用过的命题集合”
        self._last_active_props: set[str] = set()
        #耐心推进--s3
        self._patience = 8          # 连续“几乎满足”reach 的耐心步数
        self._near_eps = 0.05       # 近似阈：b_atoms>theta_up-0.05 或 sdist> -0.03
        self._reach_almost = 0

    # -------------------------
    # 标签来源统一：优先 L_stab
    # -------------------------
    def _extract_true_props(self, info: dict[str, Any]) -> set[str]:
        """优先返回稳健命题集合 L_stab；否则回退到布尔命题集合 propositions。"""
        stable = info.get('L_stab', None)
        if stable is not None:
            info['ldba_truth_source'] = 'L_stab'
            return set(stable)
        info['ldba_truth_source'] = 'propositions'
        if not self._warned_no_l_stab:
            print("[SEQ] L_stab not found in info; fallback to 'propositions'.")
            self._warned_no_l_stab = True
        return set(info.get('propositions', []))
    #耐心推进
    def _get_danger_avoid(self, info):
        # 需要 BeliefWrapper 在 info 里塞入 sdist_* （你已有）
        danger = set()
        m_avoid = getattr(self, "_m_avoid", 0.05)  # 可由构造参数传入
        for c in self.propositions:
            s = float(info.get(f"sdist_{c}", -1e9))
            if s > -m_avoid:  # 离禁区太近
                danger.add(c)
        return danger

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # 1) 环境推进 / ε-转移
        if (action == LDBASequence.EPSILON).all():
            obs, _, terminated, truncated, info = self.apply_epsilon_action()
            reward = 0.0
        else:
            assert not (action == LDBASequence.EPSILON).any()
            obs, reward, terminated, truncated, info = super().step(action)

        # 2) 当前阶段的 reach/avoid
        reach, avoid = self.goal_seq[self.num_reached]

        # —— A 分闸 ——：到达用稳定门、更鲁棒；避让用原始命题、更保守
        reach_props = set(info.get('L_stab', info.get('propositions', set())))
        avoid_props = set(info.get('propositions', set()))

        # 3) 保护带：距离禁区太近也当作“危险”（需要 BeliefWrapper 写入 sdist_*）
        avoid_props |= getattr(self, "_get_danger_avoid", lambda _info: set())(info)

        # 4) 形式化判定
        assignment_reach = Assignment({p: (p in reach_props) for p in self.propositions}).to_frozen()
        assignment_avoid = Assignment({p: (p in avoid_props) for p in self.propositions}).to_frozen()

        # 5) 先判 avoid（高优先级）
        if assignment_avoid in avoid:
            reward = -1.0
            info['violation'] = True
            terminated = True
        else:
            # 6) reach：命中 或 “耐心推进”
            reached = (reach != LDBASequence.EPSILON and assignment_reach in reach)

            # —— 准备“几乎满足”的判据（b_atoms 接近上阈，或 sdist 略为正/近0）——
            b_atoms = info.get('b_atoms', {})
            near_eps = getattr(self, "_near_eps", 0.05)      # 默认 0.05
            theta_up = getattr(self, "theta_up", 0.8)        # 若未从 BeliefWrapper 透传，则用默认
            # 把 reach 中“需要为 True 的原子命题”抽出来，避免把无关命题当作 almost
            target_true = set()
            try:
                for ass in reach:
                    for p, v in ass.items():
                        if v: target_true.add(p)
            except Exception:
                # reach 可能是 EPSILON 或其它结构，容错即可
                pass

            almost = False
            for p in target_true:
                b = float(b_atoms.get(p, 0.0))
                s = float(info.get(f"sdist_{p}", -1e9))
                if (b > (theta_up - near_eps)) or (s > -0.03):
                    almost = True
                    break

            # —— 耐心计数器：连着“几乎满足”N 步也推进 —— 
            patience = getattr(self, "_patience", 8)
            self._reach_almost = getattr(self, "_reach_almost", 0)
            self._reach_almost = (self._reach_almost + 1) if (almost and not reached) else 0

            if reached or (self._reach_almost >= patience):
                self._reach_almost = 0
                self.num_reached += 1
                terminated = self.num_reached >= len(self.goal_seq)
                if terminated:
                    info['success'] = True
                reward = 1.0 if (self.partial_reward or terminated) else 0.0

        # 7) 诊断信息（可选）
        info['prop_source'] = {
            'reach': 'L_stab' if 'L_stab' in info else 'propositions',
            'avoid': 'propositions',
            'danger_added': bool(len(avoid_props - set(info.get('propositions', set()))))
        }
        info['reach_almost_steps'] = getattr(self, "_reach_almost", 0)

        # 8) 打包返回
        self.obs = obs
        self.info = info
        obs = self.complete_observation(obs, info)
        return obs, reward, terminated, truncated, info


    def apply_epsilon_action(self):
        # 训练端同样支持 ε 推进：不消耗 env.step，只推进子目标
        assert self.goal_seq[self.num_reached][0] == LDBASequence.EPSILON
        self.num_reached += 1
        return self.obs, 0.0, False, False, self.info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.goal_seq = self.sample_sequence()
        self.num_reached = 0
        self._warned_no_l_stab = False
        self._last_active_props = set()
        obs = self.complete_observation(obs, info)
        self.obs = obs
        self.info = info
        return obs, info

    def complete_observation(self, obs: WrapperObsType, info: dict[str, Any] = None) -> WrapperObsType:
        """
        约定：
          - 训练网络只消费 'features'；
          - 'goal'/'initial_goal' 给到上层编码器；
          - 'propositions' 为底层环境的布尔集合（用于对齐原始 DeepLTL 习惯）；
          - 'propositions_used' 为本步用于判定推进的集合（list），便于记录/对比；
          - 'ldba_truth_source' 已写入 info 中（'L_stab' 或 'propositions'）。
        """
        return {
            'features': obs,
            'goal': self.goal_seq[self.num_reached:],
            'initial_goal': self.goal_seq,
            # 保留原始布尔标签，兼容旧代码/日志
            'propositions': info.get('propositions', [] if info is None else info.get('propositions', [])),
            # 新增：这一步真正用于推进的标签（稳健 or 布尔）
            'propositions_used': list(self._last_active_props),
        }
