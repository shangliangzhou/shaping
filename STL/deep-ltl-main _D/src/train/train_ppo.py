#!/usr/bin/env python
import argparse
from typing import Optional

import gymnasium
import simple_parsing
import time
import datetime

import torch

import config
import preprocessing
import torch_ac

import utils
from model.model import build_model
from envs import make_env, get_env_attr
# 训练期奖励形塑（保留）
from envs.potential_shaping_wrapper import PotentialShapingWrapper
# Phase-1：信念推进（已实现于 src/envs/belief_wrapper.py）
from envs.belief_wrapper import BeliefReachAvoidWrapper
# 命题噪声注入（漏检/误报/延迟）
from envs.proposition_noise_wrapper import PropositionNoiseWrapper

from sequence.samplers import CurriculumSampler, curricula
from utils import torch_utils
from utils.logging.file_logger import FileLogger
from utils.logging.multi_logger import MultiLogger
from utils.logging.text_logger import TextLogger
from utils.logging.wandb_logger import WandbLogger
from utils.model_store import ModelStore
from config import *


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.text_logger = TextLogger(args)
        self.model_store = ModelStore.from_config(args)

    def train(self, log_csv: bool = True, log_wandb: bool = False):
        training_status, resuming = self.get_training_status()
        envs = self.make_envs(training_status["curriculum_stage"])
        if resuming:
            self.model_store.load_vocab()
        else:
            assignments = envs[0].get_possible_assignments()
            print(f'Number of assignments: {len(assignments)}')
            preprocessing.init_vocab(assignments)
            self.model_store.save_vocab()

        model = build_model(envs[0], training_status, model_configs[self.args.model_config])
        model.to(self.args.experiment.device)

        algo = torch_ac.PPO(
            envs, model, self.args.experiment.device, self.args.ppo,
            preprocess_obss=preprocessing.preprocess_obss, parallel=False
        )

        if "optimizer_state" in training_status:
            algo.optimizer.load_state_dict(training_status["optimizer_state"])
            self.text_logger.info("Loaded optimizer from existing run.")

        logger = self.make_logger(log_csv, log_wandb, resuming)
        logger.log_config()

        self.text_logger.info(f'Num parameters: {torch_utils.get_number_of_params(model)}')

        num_steps = training_status["num_steps"]
        num_updates = training_status["num_updates"]
        num_eval_steps = training_status["num_eval_steps"]

        while num_steps < self.args.experiment.num_steps:
            # 评估与快照
            if self.args.save and (num_updates == 0 or num_eval_steps >= self.args.experiment.eval_interval):
                num_eval_steps = 0
                training_status = {"num_steps": num_steps, "num_updates": num_updates,
                                   "model_state": algo.model.state_dict()}
                self.model_store.save_eval_training_status(training_status)
                print("112")

            start = time.time()
            # 采样
            exps, logs = algo.collect_experiences()

            # ---- 形塑诊断（若 wrapper 提供可选接口）----
            try:
                if hasattr(envs[0], "get_shaping_stats_and_reset"):
                    stats = [e.get_shaping_stats_and_reset() for e in envs]
                    logs["shape_sum"]  = float(sum(s["step_shaping_sum"] for s in stats)) / max(1, len(stats))
                    logs["shape_prog"] = float(sum(s["progress_hits"]    for s in stats)) / max(1, len(stats))
            except Exception:
                pass
            # （可选）信念统计：若后续在 BeliefReachAvoidWrapper 中实现类似接口，可在此添加
            # try:
            #     if hasattr(envs[0], "get_belief_stats_and_reset"):
            #         bstats = [e.get_belief_stats_and_reset() for e in envs]
            #         logs["accept_prob_mean"] = float(sum(s["accept_prob_mean"] for s in bstats)) / max(1, len(bstats))
            #         logs["belief_entropy_mean"] = float(sum(s["belief_entropy_mean"] for s in bstats)) / max(1, len(bstats))
            # except Exception:
            #     pass
            # --------------------------------------------

            curriculum = get_env_attr(envs[0], 'sample_sequence').curriculum
            curriculum.update_task_success(logs['avg_goal_success'], verbose=True)

            # 更新
            update_logs = algo.update_parameters(exps)
            logs.update(update_logs)

            update_time = time.time() - start

            num_steps += logs["num_steps"]
            num_eval_steps += logs["num_steps"]
            num_updates += 1

            if num_updates % self.args.experiment.log_interval == 0 or curriculum.finished:
                logs = self.augment_logs(logs, update_time, num_steps)
                logger.log(logs)

            if (
                curriculum.finished
                or (self.args.experiment.save_interval > 0 and num_updates % self.args.experiment.save_interval == 0)
            ) and self.args.save:
                training_status = {
                    "num_steps": num_steps,
                    "num_updates": num_updates,
                    "model_state": algo.model.state_dict(),
                    "optimizer_state": algo.optimizer.state_dict(),
                    "curriculum_stage": curriculum.stage_index,
                    "num_eval_steps": num_eval_steps,
                }
                self.model_store.save_training_status(training_status)
                self.model_store.save_ltl_net(algo.model.ltl_net.state_dict())
                self.text_logger.info("Saved training status")
                print("520")

            if curriculum.finished:
                self.text_logger.important_info("Finished curriculum.")
                print("1314")
                break

        if self.args.save:
            try:
                curr_stage = curriculum.stage_index
            except UnboundLocalError:
                curr_stage = get_env_attr(envs[0], 'sample_sequence').curriculum.stage_index

            training_status = {
                "num_steps": num_steps,
                "num_updates": num_updates,
                "model_state": algo.model.state_dict(),
                "optimizer_state": algo.optimizer.state_dict(),
                "curriculum_stage": curr_stage,
                "num_eval_steps": num_eval_steps,
            }
            self.model_store.save_training_status(training_status)
            self.model_store.save_ltl_net(algo.model.ltl_net.state_dict())
            self.text_logger.info("Saved training status (final)")

    def make_envs(self, curriculum_stage: int) -> list[gymnasium.Env]:
        utils.set_seed(self.args.experiment.seed)
        envs = []

        for i in range(self.args.experiment.num_procs):
            curriculum = curricula[self.args.curriculum]
            curriculum.stage_index = curriculum_stage
            self.text_logger.important_info(f"Curriculum stage: {curriculum.stage_index}")
            sampler = CurriculumSampler.partial(curriculum)

            # 基础环境（含 LDBA/序列）：
            base_env = make_env(self.args.experiment.env, sampler, sequence=True)

            # 每个并行 env 的独立随机源（用于噪声 wrapper）
            env_seed = 100 * self.args.experiment.seed + i

            # 1) 命题噪声（尽量靠内层，模拟“传感层”）
            if getattr(self.args, "noise_enable", False):
                base_env = PropositionNoiseWrapper(
                    base_env,
                    p_miss=getattr(self.args, "noise_p_miss", 0.0),
                    p_false=getattr(self.args, "noise_p_false", 0.0),
                    delay_steps=getattr(self.args, "noise_delay_steps", 0),
                    seed=env_seed,
                )

            # 2) 势能塑形（保留）
            if getattr(self.args, "shaping_enable", False):
                base_env = PotentialShapingWrapper(
                    base_env,
                    beta=self.args.shaping_beta,
                    eta=self.args.shaping_eta,
                    gamma=self.args.ppo.discount,
                    alpha=getattr(self.args, "shaping_alpha", 0.0),
                )

            # 3) Phase-1：信念推进（与塑形并行，可独立开关）
            if getattr(self.args, "belief_enable", False):
                base_env = BeliefReachAvoidWrapper(
                    base_env,
                    ema_tau=getattr(self.args, "belief_ema_tau", 0.9),
                    temperature=getattr(self.args, "belief_temperature", 1.0),
                    reward_coef_delta=getattr(self.args, "belief_reward_coef_delta", 0.0),
                    expose_accept_prob_in_obs=getattr(self.args, "belief_expose_accept_prob", False),
                )

            envs.append(base_env)

        # 独立设置每个并行环境的种子（环境本身）
        seed_offset = 100 * self.args.experiment.seed
        seeds = [seed_offset + i for i in range(self.args.experiment.num_procs)]
        self.text_logger.info(f"Using seeds: {seeds}")
        for env, seed in zip(envs, seeds):
            env.reset(seed=seed)
        self.text_logger.info("Environments loaded.")
        return envs

    def get_training_status(self) -> tuple[dict, bool]:
        resuming = False
        try:
            training_status = self.model_store.load_training_status()
            self.text_logger.important_info("Resuming training from existing run.")
            resuming = True
        except FileNotFoundError:
            training_status = {"num_steps": 0, "num_updates": 0, "curriculum_stage": 0, "num_eval_steps": 0}
        return training_status, resuming

    def make_logger(self, log_csv: bool, log_wandb: bool, resuming: bool) -> MultiLogger:
        loggers = [self.text_logger]
        if log_csv:
            loggers.append(FileLogger(self.args, resuming=resuming))
        if log_wandb:
            loggers.append(WandbLogger(self.args, project_name='deep-ltl', resuming=resuming))
        return MultiLogger(*loggers)

    def augment_logs(self, logs: dict, update_time: float, num_steps: int) -> dict:
        sps = logs["num_steps"] / update_time
        remaining_duration = int((self.args.experiment.num_steps - num_steps) / sps)
        remaining_duration = 0 if remaining_duration < 0 else remaining_duration
        remaining_time = str(datetime.timedelta(seconds=remaining_duration))

        average_reward_per_step = utils.average_reward_per_step(
            logs["return_per_episode"], logs["num_steps_per_episode"]
        )
        average_discounted_return = utils.average_discounted_return(
            logs["return_per_episode"], logs["num_steps_per_episode"], self.args.ppo.discount
        )
        logs.update({
            "arps": average_reward_per_step,
            "adr": average_discounted_return,
            'sps': sps,
            'remaining': remaining_time,
            'num_steps': num_steps
        })
        return logs


# noinspection PyTypeChecker
def parse_arguments() -> argparse.Namespace:
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(config.ExperimentConfig, dest="experiment")
    parser.add_arguments(config.PPOConfig, dest="ppo")
    parser.add_argument("--model_config", type=str, default="default", choices=model_configs.keys(),
                        required=True)
    parser.add_argument("--curriculum", type=str, choices=curricula.keys(), required=True)
    parser.add_argument("--log_csv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, default=True)

    # --- shaping args ---
    parser.add_argument("--shaping.enable", dest="shaping_enable",
                        action="store_true", default=False,
                        help="Enable potential-based reward shaping")
    parser.add_argument("--shaping.beta", dest="shaping_beta",
                        type=float, default=0.01,
                        help="beta in shaping term (recommend ~5*(1-gamma))")
    parser.add_argument("--shaping.eta", dest="shaping_eta",
                        type=float, default=0.02,
                        help="tiny bonus when remain decreases")
    parser.add_argument("--shaping.alpha", dest="shaping_alpha",
                        type=float, default=0.5,
                        help="weight for next-subgoal matching distance in Phi(s)")

    # --- belief args (Phase-1) ---
    parser.add_argument("--belief.enable", dest="belief_enable",
                        action="store_true", default=False,
                        help="Enable proposition belief & expected pulse shaping")
    parser.add_argument("--belief.ema_tau", dest="belief_ema_tau",
                        type=float, default=0.9,
                        help="EMA smoothing for proposition probabilities (0~1)")
    parser.add_argument("--belief.temperature", dest="belief_temperature",
                        type=float, default=1.0,
                        help="Temperature scaling for probability calibration (>0)")
    parser.add_argument("--belief.reward_coef_delta", dest="belief_reward_coef_delta",
                        type=float, default=0.0,
                        help="Coefficient for expected event-pulse via accept_prob delta")
    parser.add_argument("--belief.expose_accept_prob", dest="belief_expose_accept_prob",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="If true, append accept_prob into observation dict")

    # --- proposition noise args ---
    parser.add_argument("--noise_enable", dest="noise_enable",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Enable proposition noise injection (miss/false/delay)")
    parser.add_argument("--noise_p_miss", dest="noise_p_miss",
                        type=float, default=0.0,
                        help="Probability of miss detection (true->false)")
    parser.add_argument("--noise_p_false", dest="noise_p_false",
                        type=float, default=0.0,
                        help="Probability of false alarm (false->true)")
    parser.add_argument("--noise_delay_steps", dest="noise_delay_steps",
                        type=int, default=0,
                        help="FIFO delay (in steps) for propositions")

    args = parser.parse_args()

    if args.experiment.device == 'gpu':
        assert torch.cuda.is_available(), "CUDA is not available."
        args.experiment.device = 'cuda'

    return args


def main():
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    args = parse_arguments()
    trainer = Trainer(args)
    start_time = time.time()
    trainer.train(log_csv=args.log_csv, log_wandb=args.log_wandb)
    training_time = datetime.timedelta(seconds=int(time.time() - start_time))
    print(f"Training took {training_time}.")


# 运行示例（含噪声与信念推进）：
# python run_zones.py --device cuda --name ppo_belief_eta002 --seed 1 \
#   --shaping_enable true --shaping_beta 0.01 --shaping_eta 0.02 --shaping_alpha 0.5 \
#   --belief_enable true --belief_ema_tau 0.9 --belief_temperature 1.0 \
#   --belief_reward_coef_delta 0.02 --belief_expose_accept_prob true \
#   --noise_enable true --noise_p_miss 0.05 --noise_p_false 0.02 --noise_delay_steps 1 \
#   --model_config PointLtl2-v0 --curriculum PointLtl2-v0
if __name__ == '__main__':
    main()
