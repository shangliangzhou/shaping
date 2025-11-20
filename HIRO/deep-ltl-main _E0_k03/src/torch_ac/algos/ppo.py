from typing import Callable
import time
import numpy
import torch

from config import PPOConfig
from torch_ac.algos.base import BaseAlgo
import math

class PPO(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, model, device, config: PPOConfig, preprocess_obss: Callable, parallel=False):

        num_steps_per_proc = config.steps_per_process

        super().__init__(envs, model, device, num_steps_per_proc, config.discount, config.lr, config.gae_lambda,
                         config.entropy_coef, config.value_loss_coef, config.max_grad_norm, preprocess_obss, parallel=parallel)

        self.clip_eps = config.clip_eps
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.act_shape = envs[0].action_space.shape

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.model.parameters(), config.lr, eps=config.optim_eps)
        self.batch_num = 0

    def update_parameters(self, exps):
        # Collect experiences
        #每个epoch结束后，通常，测试一次性能，保存一次模型
        for epoch_idx in range(self.epochs):   #遍历每一个epoch ， 一个epoch表示模型完整训练一次，训练一般会进行多次epoch
            # Initialize log values
            epoch_t0 = time.perf_counter()   # ← 开始计时

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            #一个batch表示一次迭代，一次迭代包含一次前向+反向传播更新一次参数（输出loss，进行剃度下降更新参数）
            for inds in self._get_batches_starting_indexes(): # ← 这一层是 mini-batch 循环（每次一次 optimizer.step()）
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                if self.model.recurrent:
                    memory = exps.memory[inds]

                ## === 序列展开（recurrence）===
                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # Compute loss

                    if self.model.recurrent:
                        dist, value, memory = self.model(sb.obs, memory * sb.mask)
                    else:
                        dist, value = self.model(sb.obs)

                    entropy = dist.entropy().mean()

                    # ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    delta_log_prob = dist.log_prob(sb.action) - sb.log_prob
                    # if (len(self.act_shape) == 1):  # Not scalar actions (multivariate)
                    #    delta_log_prob = torch.sum(delta_log_prob, dim=1)
                    ratio = torch.exp(delta_log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss
                    if loss.isnan():
                        print("Loss is NaN")

                    # Update batch values
                
                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch
            
                    if self.model.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values
                #归一化
                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic
                # 更新一次
                self.optimizer.zero_grad()
                batch_loss.backward()
                # p.grad can be None if the GNN is not used (e.g. because all assignments only involve a single proposition)
                grad_norm = sum(
                    p.grad.data.norm(2).item() ** 2 for p in self.model.parameters() if p.requires_grad and p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad and p.grad is not None],
                                               self.max_grad_norm)
                self.optimizer.step()

                if any(torch.isnan(p).any() for p in self.model.parameters()):
                    print("Model parameters are NaN")

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

             # ← 本 epoch 结束处打印耗时（以及该 epoch 的 mini-batch 数）
            n_batches = math.ceil(self.num_steps / self.batch_size)
            print(f"[PPO] epoch {epoch_idx+1}/{self.epochs} took {time.perf_counter() - epoch_t0:.3f}s "
                f"(mini-batches: {n_batches}, batch_size: {self.batch_size}, recurrence: {self.recurrence})")
        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms)
        }

        return logs

    def _get_batches_starting_indexes(self):
        """"
            为每个批次提供首次给模型的观测值索引和用于计算损失的经验索引。

            首先，索引是从0到`self.num_frames`的整数，步长为`self.recurrence`，
            每隔一次将起始索引偏移`self.recurrence//2`，以获得更加多样化的批次。
            然后，将这些索引分割成不同的批次。

            返回
            -------
            batches_starting_indexes : list of list of int
                每个批次首次使用的经验索引
        """

        indexes = numpy.arange(0, self.num_steps, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_steps_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
