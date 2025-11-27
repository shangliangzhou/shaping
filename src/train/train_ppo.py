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

        num_steps = training_status["num_steps"]           # 总环境步数
        num_updates = training_status["num_updates"]       # 参数更新次数
        num_eval_steps = training_status["num_eval_steps"] # 评估间隔内的环境步数

        while num_steps < self.args.experiment.num_steps:
            # 指定每隔多少个环境步进行一次评估保存
            if self.args.save and (num_updates == 0 or num_eval_steps >= self.args.experiment.eval_interval):
                num_eval_steps = 0
                training_status = {"num_steps": num_steps, "num_updates": num_updates,
                                   "model_state": algo.model.state_dict()}
                self.model_store.save_eval_training_status(training_status)

            start = time.time()
            # 每次 rollout 采样
            exps, logs = algo.collect_experiences()

            # （已关闭奖励塑形，不做形塑相关统计）

            curriculum = get_env_attr(envs[0], 'sample_sequence').curriculum
            curriculum.update_task_success(logs['avg_goal_success'], verbose=True)

            # PPO 参数更新
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

            if curriculum.finished:
                self.text_logger.important_info("Finished curriculum.")
                break

        if self.args.save:
            # 取 curriculum 当前阶段，避免未定义
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

            # === 训练端环境：Sequence 训练 + 信念化（L_stab） ===
            base_env = make_env(
                self.args.experiment.env,
                sampler,
                sequence=True,              # 训练使用 reach-avoid 序列
                # —— 训练端不加噪声（C→C） ——
                eval_noise_enable=False,
                # —— 信念化开关与超参（来自命令行） ——
                belief_enable=self.args.belief_enable,
                belief_theta_up=self.args.belief_theta_up,
                belief_theta_down=self.args.belief_theta_down,
                belief_k_stab=self.args.belief_k_stab,
                belief_ema_lambda=self.args.belief_ema_lambda,
                belief_k=self.args.belief_k,
                belief_m=self.args.belief_m,
            )

            envs.append(base_env)

        # 独立设置每个并行环境的种子，避免重叠
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
            'num_steps': num_steps  # set num_steps to the total number of steps
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

    # ===== 信念化（BeliefWrapper）超参（训练端建议开启）=====
    parser.add_argument("--belief.enable", dest="belief_enable",
                        action="store_true", default=True,
                        help="Enable belief-based proposition smoothing + stability gate")
    parser.add_argument("--belief.theta_up", dest="belief_theta_up",
                        type=float, default=0.8, help="upper threshold for hardening belief to True")
    parser.add_argument("--belief.theta_down", dest="belief_theta_down",
                        type=float, default=0.2, help="lower threshold for hardening belief to False")
    parser.add_argument("--belief.k_stab", dest="belief_k_stab",
                        type=int, default=3, help="stable frames required for acceptance into L_stab")
    parser.add_argument("--belief.ema_lambda", dest="belief_ema_lambda",
                        type=float, default=0.8, help="EMA factor for per-step proposition probabilities")
    parser.add_argument("--belief.k", dest="belief_k",
                        type=float, default=5.0, help="sdist->prob temperature slope")
    parser.add_argument("--belief.m", dest="belief_m",
                        type=float, default=0.0, help="safety margin added to signed distance before squashing")

    # ==== 奖励塑形（已在本阶段关闭：不再提供命令行开关）====

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


# 运行示例（见 run_zones.py）
if __name__ == '__main__':
    main()
