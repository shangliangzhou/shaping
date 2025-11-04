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
# [ADD] import the training-time shaping wrapper
from envs.potential_shaping_wrapper import PotentialShapingWrapper  # <-- 新增

from sequence.samplers import CurriculumSampler, curricula
from utils import torch_utils
from utils.logging.file_logger import FileLogger
from utils.logging.multi_logger import MultiLogger
from utils.logging.text_logger import TextLogger
from utils.logging.wandb_logger import WandbLogger
from utils.model_store import ModelStore
from config import *
# （你之前加的） from train.reward_shaping import shape_episode_rewards  # <-- 本方案不再需要，可移除
# 顶部 import
from train.her_suffix import split_episodes_from_batch, suffix_her_episode, aux_value_update
from envs.episode_buffer_wrapper import EpisodeBufferWrapper



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
        algo = torch_ac.PPO(envs, model, self.args.experiment.device, self.args.ppo,
                            preprocess_obss=preprocessing.preprocess_obss, parallel=False)
        if "optimizer_state" in training_status:
            algo.optimizer.load_state_dict(training_status["optimizer_state"])
            self.text_logger.info("Loaded optimizer from existing run.")
        logger = self.make_logger(log_csv, log_wandb, resuming)
        logger.log_config()

        self.text_logger.info(f'Num parameters: {torch_utils.get_number_of_params(model)}')
        num_steps = training_status["num_steps"] #总环境步数
        num_updates = training_status["num_updates"] #参数更新次数
        num_eval_steps = training_status["num_eval_steps"] #评估间隔内的环境步数
        while num_steps < self.args.experiment.num_steps:
            #指定每隔多少个环境步数进行一次模型评估
            if self.args.save and (num_updates == 0 or num_eval_steps >= self.args.experiment.eval_interval):
                num_eval_steps = 0
                training_status = {"num_steps": num_steps, "num_updates": num_updates,
                                   "model_state": algo.model.state_dict()}
                self.model_store.save_eval_training_status(training_status)
                print("112")
            start = time.time()
            #表示每个进程在一次rollot采样周期，在环境中执行的步数
            exps, logs = algo.collect_experiences()
            # ---- 形塑诊断统计（若 wrapper 提供该接口）----
            try:
                if hasattr(envs[0], "get_shaping_stats_and_reset"):
                    stats = [e.get_shaping_stats_and_reset() for e in envs]
                    # 平均到每个并行环境，写入日志
                    logs["shape_sum"]  = float(sum(s["step_shaping_sum"] for s in stats)) / max(1, len(stats))
                    logs["shape_prog"] = float(sum(s["progress_hits"]    for s in stats)) / max(1, len(stats))
            except Exception as _e:
                # 静默失败即可；不影响训练
                pass
# --------------------------------------------

            curriculum = get_env_attr(envs[0], 'sample_sequence').curriculum
            curriculum.update_task_success(logs['avg_goal_success'], verbose=True)

            # [保持原样] 不改 PPO 算法与优势/回报：奖励形塑已在 env wrapper 内完成
            update_logs = algo.update_parameters(exps)
            logs.update(update_logs)

            if getattr(self.args, "her_enable", False):
                try:
                    raw_episodes = []
                    for e in envs:
                        if hasattr(e, "pop_recent_episodes"):
                            raw_episodes.extend(e.pop_recent_episodes())

                    her_ready_eps = []
                    for ep in raw_episodes:
                        obss = ep["obss"]
                        acts = ep["acts"]
                        # 用 0 作为基线，后续用势函数重算（更干净）
                        her_ready_eps.append({"obss": obss, "acts": acts, "rews_env": [0.0] * len(acts)})

                    her_eps = []
                    for ep in her_ready_eps:
                        her_eps.extend(suffix_her_episode(ep, k_future=self.args.her_k_future))

                    her_logs = {}
                    if len(her_eps) > 0:
                        her_logs = aux_value_update(
                            model=algo.model,
                            optimizer=algo.optimizer,
                            preprocess_obss=preprocessing.preprocess_obss,
                            her_eps=her_eps,
                            gamma=self.args.ppo.discount,
                            beta=self.args.shaping_beta if getattr(self.args, "shaping_enable", False) else 0.0,
                            eta=self.args.shaping_eta  if getattr(self.args, "shaping_enable", False) else 0.0,
                            alpha=getattr(self.args, "shaping_alpha", 0.0),
                            batch_size=self.args.her_aux_batch,
                            epochs=self.args.her_aux_epochs,
                            device=self.args.experiment.device,
                            lambda_bc=self.args.her_lambda_bc,
                        )
                    logs["her_used"]  = her_logs.get("her_used", 0)
                    logs["her_vloss"] = her_logs.get("her_vloss", 0.0)
                except Exception as e:
                    self.text_logger.info(f"[HER] skipped due to error: {e}")






            update_time = time.time() - start

            num_steps += logs["num_steps"]
            num_eval_steps += logs["num_steps"]
            num_updates += 1
            if num_updates % self.args.experiment.log_interval == 0 or curriculum.finished:
                logs = self.augment_logs(logs, update_time, num_steps)
                logger.log(logs)
            if (
                    curriculum.finished or self.args.experiment.save_interval > 0 and num_updates % self.args.experiment.save_interval == 0) \
                    and self.args.save:
                training_status = {"num_steps": num_steps, "num_updates": num_updates,
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
        # for i in range(self.args.experiment.num_procs):
        #     curriculum = curricula[self.args.curriculum]
        #     curriculum.stage_index = curriculum_stage
        #     self.text_logger.important_info(f"Curriculum stage: {curriculum.stage_index}")
        #     sampler = CurriculumSampler.partial(curriculum)

        #     # [修改点] 先创建基础 env
        #     base_env = make_env(self.args.experiment.env, sampler, sequence=True)

        #     # [新增包裹] 仅在训练、且显式开启开关时启用势函数形塑
        #     if self.args.shaping_enable:
        #         base_env = PotentialShapingWrapper(
        #             base_env,
        #             beta=self.args.shaping_beta,
        #             eta=self.args.shaping_eta,
        #             gamma=self.args.ppo.discount,
        #             alpha=self.args.shaping_alpha    # <-- 新增
        #         )

        #     envs.append(base_env)

        for i in range(self.args.experiment.num_procs):
            curriculum = curricula[self.args.curriculum]
            curriculum.stage_index = curriculum_stage
            self.text_logger.important_info(f"Curriculum stage: {curriculum.stage_index}")
            sampler = CurriculumSampler.partial(curriculum)

            base_env = make_env(self.args.experiment.env, sampler, sequence=True)

            # 你之前的奖励形塑wrapper（若启用）
            if getattr(self.args, "shaping_enable", False):
                from envs.potential_shaping_wrapper import PotentialShapingWrapper
                base_env = PotentialShapingWrapper(
                    base_env,
                    beta=self.args.shaping_beta,
                    eta=self.args.shaping_eta,
                    gamma=self.args.ppo.discount,
                    alpha=getattr(self.args, "shaping_alpha", 0.0),
                )

            # >>> 新增：HER记录器（放在最外层，拿到“原始obs”） <<<
            if getattr(self.args, "her_enable", False):
                base_env = EpisodeBufferWrapper(base_env)

            envs.append(base_env)

        # Set different seeds for each environment. The seed offset is used to ensure that the seeds do not overlap.
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

        average_reward_per_step = utils.average_reward_per_step(logs["return_per_episode"],
                                                                logs["num_steps_per_episode"])
        average_discounted_return = utils.average_discounted_return(logs["return_per_episode"],
                                                                    logs["num_steps_per_episode"],
                                                                    self.args.ppo.discount)
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

#    # --- ADD: shaping args (use dest to normalize names) ---
#     parser.add_argument("--shaping.enable", dest="shaping_enable",
#                         action="store_true", default=False,
#                         help="Enable potential-based reward shaping")
#     parser.add_argument("--shaping.beta", dest="shaping_beta",
#                         type=float, default=1.0,
#                         help="beta for shaping term")
#     parser.add_argument("--shaping.eta", dest="shaping_eta",
#                         type=float, default=0.0,
#                         help="tiny bonus for subgoal completion (0 to disable)")
#     # --- END ADD ---
    # --- ADD: shaping args ---
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
    # --- END ADD ---
    # parse_arguments() 里追加（紧跟着 shaping 参数）
    parser.add_argument("--her.enable", dest="her_enable",
                        action="store_true", default=False,
                        help="Enable suffix-HER for aux value update (safe for PPO)")
    parser.add_argument("--her.k_future", dest="her_k_future",
                        type=int, default=4, help="Future indices sampled per episode for HER")
    parser.add_argument("--her.aux_epochs", dest="her_aux_epochs",
                        type=int, default=1, help="Aux value update epochs per PPO update")
    parser.add_argument("--her.aux_batch", dest="her_aux_batch",
                        type=int, default=1024, help="Aux value update batch size")
    parser.add_argument("--her.lambda_bc", dest="her_lambda_bc",
                        type=float, default=0.0, help="Optional tiny BC weight on HER samples")



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


# python run_zones.py --device cuda --name ppo_shaping_beta001_eta002 --seed 1 \
#   --shaping_enable true --shaping_beta 0.01 --shaping_eta 0.02
if __name__ == '__main__':
    main()
