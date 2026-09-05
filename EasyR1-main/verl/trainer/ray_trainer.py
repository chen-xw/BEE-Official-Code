# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type, List

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)

import time
import datetime
from .data_loader import create_filtered_train_dataloader
from ..utils.dataset import RLHFDataset, collate_fn, FiltereRLHFdDataset
from mathruler.grader import extract_boxed_content, grade_answer

from tools.api_judge import api_batch_judge, rule_based_match
from tools.custom_api import build_deepseek_client, build_gemini_client, build_qwen_client

from rich import print as rich_print
import datetime
def safe_print(*args, **kwargs):
    # 获取环境变量 RANK，默认为 "0"。
    # 如果是通过 torchrun 启动，RANK 会被自动设置。
    # 如果是普通的 python 启动，RANK 默认为 0，会正常打印。
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:  
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_tag = f"[bold cyan][{timestamp}][/bold cyan]"
        rich_print(time_tag, *args, **kwargs)


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        #--------------------------New-----------------------------------------#
        base_dataset: RLHFDataset = None,   
        bootstrap_train_dataloder: Optional[StatefulDataLoader] = None,
        bootstrap_val_dataloader: Optional[StatefulDataLoader] = None,
        #--------------------------New-----------------------------------------#
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        #--------------------------New-----------------------------------------#
        self.difficulty_dict_train = {}
        self.target_high_entropy_token_num_dict = {}
        # Adaptive entropy shaping alpha per difficulty
        def _get_conf(name, default):
            return getattr(self.config, name, default)
        self.alpha_entropy_lr = _get_conf("alpha_entropy_lr", 0.05)
        self.alpha_entropy_min = _get_conf("alpha_entropy_min", 0.0)
        self.alpha_entropy_max = _get_conf("alpha_entropy_max", 2.0)
        self.alpha_entropy_dict = {
            "easy": _get_conf("alpha_entropy_init_easy", 0.5),
            "medium": _get_conf("alpha_entropy_init_medium", 0.5),
            "hard": _get_conf("alpha_entropy_init_hard", 0.5),
        }
        self.skip_gid_set = set()
        self.bootstrap_train_dataloader = bootstrap_train_dataloder
        self.bootstrap_val_dataloader = bootstrap_val_dataloader
        self.base_dataset = base_dataset
        self.epoch_update_iter = 0
        self.epoch_id = 0
        self.drop_allpass_entropy_threshold = -1
        self.high_entropy_token_threshold = -1
        #--------------------------New-----------------------------------------#

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

        if "api" in self.config.worker.rule_based_judge.judge_function_name:
            if self.config.worker.rule_based_judge.api_name in ['deepseek-chat', 'deepseek']:
                self.client = build_deepseek_client()
            elif self.config.worker.rule_based_judge.api_name == 'gemini-2.5-pro':
                self.client = build_gemini_client()
            elif self.config.worker.rule_based_judge.api_name in  ['Qwen2.5-VL-32B', 'Qwen3-VL-235B-A22B-Instruct', 'Qwen3.5-397B-A17B', 'Qwen3-VL-4B']:
                self.client = build_qwen_client()
            else:
                self.client = None
                raise ValueError(f"API {self.config.worker.rule_based_judge.api_name} not supported.")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)
    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        # Per-sample detail records for Excel export
        val_detail_records = []
        reward_metrics_lst = defaultdict(list)
        uid2rollouts = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "problem"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps
            test_gen_batch.meta_info["high_entropy_threshold"] = self.high_entropy_token_threshold

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)


            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

            # xiuwei
            messages_batch = test_batch.non_tensor_batch['messages']

            problems = []
            # 2. 遍历 Batch 中的每一条数据
            for messages in messages_batch:
                current_problem = ""
                for msg in messages:
                    if msg['role'] == 'user':
                        raw_content = msg['content']
                        # 清洗并提取纯问题
                        clean_text = raw_content.replace('<image>\n', '')
                        current_problem = clean_text #.split('\n')[0]
                        break
                problems.append(current_problem)

            # 3. 极其关键：将列表转回 numpy 数组，并指定 dtype=object
            # 这样才能完美兼容 veRL 底层 DataProto 的规范
            test_batch.non_tensor_batch['problem'] = np.array(problems, dtype=object)
            
            # evaluate using reward_function
            if 'api' in self.config.worker.rule_based_judge.judge_function_name:
                #breakpoint()
                correctness_list = api_batch_judge(
                    questions=test_batch.non_tensor_batch["problem"].tolist(),
                    preds=output_texts,
                    gts=test_batch.non_tensor_batch["ground_truth"].tolist(),
                    api_name=self.config.worker.rule_based_judge.api_name,
                    api_kwargs=self.config.worker.rule_based_judge.api_kwargs,
                    client=self.client,
                    repetition_penalty=self.config.worker.reward.repetition_penalty,
                )
                #correctness_list = ray.get(self.rule_based_judge.judge.remote(output_texts, test_batch.non_tensor_batch["ground_truth"].tolist()))
                test_batch.non_tensor_batch["correctness"] = correctness_list

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            accuracies = reward_metrics["accuracy"]
            lengths = test_batch.batch["response_mask"].sum(dim=-1).cpu().tolist()
            global_ids = test_batch.non_tensor_batch["global_id"]
            entropies = test_batch.non_tensor_batch["entropies"]
            high_entropy_token_num = test_batch.non_tensor_batch["high_entropy_token_num"]

            for gid, acc, entropy,high_ety_token_num, length in zip(global_ids, accuracies,entropies,high_entropy_token_num, lengths):
                uid2rollouts[gid].append({'length': length,'entropy':entropy, "high_ety_token_num":high_ety_token_num,'is_correct': acc > 0.5})
            

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            # Collect per-sample detail for Excel export
            ground_truths = test_batch.non_tensor_batch["ground_truth"].tolist()
            for i in range(len(scores)):
                is_correct = 1 if accuracies[i] > 0.5 else 0
                val_detail_records.append({
                    "question": problems[i],
                    "answer": ground_truths[i],
                    "output": output_texts[i],
                    "correct": is_correct,
                    "score": scores[i],
                    "accuracy": accuracies[i],
                    "global_id": str(global_ids[i]) if i < len(global_ids) else "",
                })

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()

        # 
        if self.high_entropy_token_threshold < 0:
            self.high_entropy_token_threshold = np.mean(test_batch.non_tensor_batch["entropy_percentile_thresholds"])
        difficulty_dict_val = {}
        difficulty_rollouts = defaultdict(list)

        for gid, rollouts in uid2rollouts.items():
            acc_rate = sum(r["is_correct"] for r in rollouts) / len(rollouts)
            if acc_rate >= 2 / 3:
                diff = "easy"
            elif acc_rate >= 1 / 3:
                diff = "medium"
            else:
                diff = "hard"
            difficulty_dict_val[gid] = diff
            difficulty_rollouts[diff].extend(rollouts)

        self.difficulty_dict_val = difficulty_dict_val
        target_high_entropy_num_dict = {}

        all_high_entropy_token_num = []
        for rollouts in difficulty_rollouts.values():
            all_high_entropy_token_num.extend([r["high_ety_token_num"] for r in rollouts])

        min_entropy = np.min(all_high_entropy_token_num)
        max_entropy = np.max(all_high_entropy_token_num)

        difficulty_entropy_coef = {"easy": 0.35, "medium": 0.5, "hard": 0.65}

        for diff in ["easy", "medium", "hard"]:
            target_high_entropy_num_dict[diff] = int(min_entropy + difficulty_entropy_coef[diff] * (max_entropy - min_entropy))

        for diff in ["easy", "medium", "hard"]:
            rollouts = difficulty_rollouts[diff]
            
            # calculate expected_entropy
            entropies = [r["entropy"] for r in rollouts]
            
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        # val_reward_metrics.update({"val/self.expected_high_entropy_token_num": self.target_high_entropy_token_num_dict})
        val_reward_metrics.update({f"val/expected_high_entropy_token_num/{diff}": val for diff, val in self.target_high_entropy_token_num_dict.items()})
        val_reward_metrics.update({"val/self.high_entropy_token_threshold": self.high_entropy_token_threshold})

        # Save per-sample validation results to Excel when val_save_details=True
        if self.config.trainer.val_save_details and val_detail_records:
            self._save_val_results_to_excel(val_detail_records)

        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _save_val_results_to_excel(self, records: list[dict]) -> None:
        import pandas as pd
        output_path = self.config.trainer.val_output_path
        if output_path is None:
            output_path = os.path.join(self.config.trainer.save_checkpoint_path, "val_results.xlsx")
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        df = pd.DataFrame(records)
        # Sort: incorrect first so errors are easy to find
        df = df.sort_values(by="correct", ascending=True).reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "index"
        df.to_excel(output_path, engine="openpyxl")
        total = len(df)
        correct = df["correct"].sum()
        safe_print(f"Validation results saved to {output_path} ({correct}/{total} correct, accuracy={correct/total:.4f})")

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
                "high_entropy_threshold": self.high_entropy_token_threshold,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )

            # xiuwei
            messages_batch = new_batch.non_tensor_batch['messages']

            problems = []
            # 2. 遍历 Batch 中的每一条数据
            for messages in messages_batch:
                current_problem = ""
                for msg in messages:
                    if msg['role'] == 'user':
                        raw_content = msg['content']
                        # 清洗并提取纯问题
                        clean_text = raw_content.replace('<image>\n', '')
                        current_problem = clean_text #.split('\n')[0]
                        break
                problems.append(current_problem)

            # 3. 极其关键：将列表转回 numpy 数组，并指定 dtype=object
            # 这样才能完美兼容 veRL 底层 DataProto 的规范
            new_batch.non_tensor_batch['problem'] = np.array(problems, dtype=object)

            # pop those keys for generation
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps", "high_entropy_threshold"],
            )

            # generate a batch
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            # repeat to align with repeated responses in rollout
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            # after all samples in the batch are judged, use ray.get to get the results
            judge_results_and_answer_texts_proto = self.actor_rollout_ref_wg.compute_rule_based_judge(data=new_batch)
            judge_results, answer_texts = judge_results_and_answer_texts_proto.non_tensor_batch["correctness"], judge_results_and_answer_texts_proto.non_tensor_batch["response_strs"]
            new_batch.non_tensor_batch["correctness"] = judge_results

            # filter group
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No sample is kept after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    def bootstrap(self):
        """
        Bootstrap difficulty_dict and target_length_dict before training using rollouts.
        """
        print("[Bootstrap] Start bootstrap epoch for Laser-D difficulty + length estimation.")
        rollout_n = self.config.worker.rollout.n
        # max_batches = getattr(self.config.trainer, "bootstrap_max_batches", 20)

        uid2acc = defaultdict(list)
        uid2len = defaultdict(list)

        self.actor_rollout_ref_wg.prepare_rollout_engine()

        self.data_iterator = iter(self.bootstrap_train_dataloader)

        # self.actor_rollout_ref_wg.release_rollout_engine()
        for batch_dict in self.bootstrap_train_dataloader:
            try:
                # construct DataProto
                test_batch = DataProto.from_single_dict(batch_dict)
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                )
                test_gen_batch.meta_info = {
                    "n": rollout_n,
                    "min_pixels": self.config.data.min_pixels,
                    "max_pixels": self.config.data.max_pixels,
                    "video_fps": self.config.data.video_fps,
                }

                test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)

                test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)

                test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * rollout_n)

                # concatnate to final result
                test_batch = test_batch.repeat(repeat_times=rollout_n, interleave=True)
                test_batch = test_batch.union(test_output_gen_batch)

                # xiuwei
                messages_batch = test_batch.non_tensor_batch['messages']

                problems = []
                # 2. 遍历 Batch 中的每一条数据
                for messages in messages_batch:
                    current_problem = ""
                    for msg in messages:
                        if msg['role'] == 'user':
                            raw_content = msg['content']
                            # 清洗并提取纯问题
                            clean_text = raw_content.replace('<image>\n', '')
                            current_problem = clean_text #.split('\n')[0]
                            break
                    problems.append(current_problem)

                # 3. 极其关键：将列表转回 numpy 数组，并指定 dtype=object
                # 这样才能完美兼容 veRL 底层 DataProto 的规范
                test_batch.non_tensor_batch['problem'] = np.array(problems, dtype=object)

                # after all samples in the batch are judged, use ray.get to get the results
                judge_results_and_answer_texts_proto = self.actor_rollout_ref_wg.compute_rule_based_judge(data=test_batch)
                judge_results, answer_texts = judge_results_and_answer_texts_proto.non_tensor_batch["correctness"], judge_results_and_answer_texts_proto.non_tensor_batch["response_strs"]
                test_batch.non_tensor_batch["correctness"] = judge_results
                
                # evaluate reward
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(test_batch))
                accuracies = reward_metrics["accuracy"]
                lengths = test_batch.batch["response_mask"].sum(dim=-1).cpu().tolist()
                global_ids = test_batch.non_tensor_batch["global_id"]
                entropies = test_batch.non_tensor_batch["entropies"]

                # accumulate per uid statistics
                # inject per-sample alpha_entropy resolved from difficulty
                test_batch.non_tensor_batch["accuracy"] = accuracies
                for gid, acc, L in zip(global_ids, accuracies, lengths):
                    uid2acc[gid].append(float(acc))
                    uid2len[gid].append(int(L))

                self.update_difficulty_and_skip_gid_set(test_batch)

            except Exception as e:
                print(f"[Bootstrap] Failed to process batch: {e}")
                continue

        self.actor_rollout_ref_wg.release_rollout_engine()

        # assign initial difficulty buckets by mean accuracy over rollouts
        difficulty_dict_bootstrap = {}
        bucket_acc, bucket_len, bucket_cnt = defaultdict(list), defaultdict(list), defaultdict(int)
        for gid, acc_list in uid2acc.items():
            acc_rate = float(np.mean(acc_list)) if len(acc_list)>0 else 0.0
            if not hasattr(self, "bootstrap_acc_dict"):                                                                                                      
                self.bootstrap_acc_dict = {}                                                                                                                 
            self.bootstrap_acc_dict[gid] = float(acc_rate)
            if acc_rate >= 2/3:
                diff = "easy"
            elif acc_rate >= 1/3:
                diff = "medium"
            else:
                diff = "hard"
            difficulty_dict_bootstrap[gid] = diff
            L_list = uid2len.get(gid, [])
            if len(L_list)>0:
                bucket_acc[diff].append(acc_rate)
                bucket_len[diff].append(float(np.mean(L_list)))
                bucket_cnt[diff] += 1

        # persist bootstrap difficulty mapping
        out_dir = os.path.join(self.config.trainer.load_checkpoint_path or self.config.trainer.experiment_name, "bootstrap")
        try:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "difficulty_bootstrap.json"), "w") as f:
                json.dump(difficulty_dict_bootstrap, f)
        except Exception as e:
            print(f"[Bootstrap] Failed to save difficulty mapping: {e}")

        # seed training difficulty dict with bootstrap mapping
        self.difficulty_dict_train.update(difficulty_dict_bootstrap)

        # log per-bucket baseline stats
        baseline_metrics = {}
        # store baseline for progress tracking
        self.bootstrap_baseline = {}
        for diff in ("easy","medium","hard"):
            if bucket_cnt[diff] > 0:
                baseline_metrics[f"bootstrap/{diff}/count"] = int(bucket_cnt[diff])
                baseline_metrics[f"bootstrap/{diff}/acc_mean"] = float(np.mean(bucket_acc[diff]))
                baseline_metrics[f"bootstrap/{diff}/len_mean"] = float(np.mean(bucket_len[diff]))
                self.bootstrap_baseline[diff] = {
                    "acc_mean": float(np.mean(bucket_acc[diff])),
                    "len_mean": float(np.mean(bucket_len[diff])),
                }
        self.logger.log(data=baseline_metrics, step=self.global_step)

        self.logger.log(
            data={
                "epoch/id": self.epoch_id,
                "epoch/skip_gid_count": len(self.skip_gid_set),
                "epoch/iter_num":len(self.train_dataloader) 
            },
            step=self.global_step
        )


    def compute_batch_accuracy(self, batch: DataProto) -> List[float]:
        # Fallback: only used when correctness is not available in batch
        responses = batch.batch["responses"]
        response_lengths = batch.batch["response_mask"].sum(dim=-1)
        problems = batch.non_tensor_batch.get("problem", [None] * len(responses)) 
        ground_truths = batch.non_tensor_batch["ground_truth"]
        accuracies = []

        for response_ids, gt, response_length in zip(responses, ground_truths, response_lengths):
            valid_response_ids = response_ids[: response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=True
            )
            # Use the unified rule_based_match from api_judge
            match_result = rule_based_match(response_str, gt)
            if match_result is True:
                accuracies.append(1.0)
            elif match_result is False:
                # Try math-based grading as secondary fallback
                answer = self._extract_answer_from_response(response_str)
                is_correct = grade_answer(answer, gt)
                accuracies.append(1.0 if is_correct else 0.0)
            else:
                # None — cannot determine rule-based, conservatively mark 0
                # (In practice, api_batch_judge should be used instead)
                accuracies.append(0.0)

        return accuracies
    
    @staticmethod
    def _exact_match(pred: str, gt: str) -> bool:
        """Delegate to the unified rule_based_match from api_judge."""
        result = rule_based_match(pred, gt)
        return result is True
    
    @staticmethod
    def _extract_answer_from_response(response_str: str) -> str:
        """Extract answer from <answer>...</answer> tags, same as reward function."""
        from tools.api_judge import extract_answer_from_tags
        result = extract_answer_from_tags(response_str)
        if result:
            return result
        return extract_boxed_content(response_str)
    
    def update_difficulty_and_skip_gid_set(self, batch: DataProto) -> None:
        global_ids = batch.non_tensor_batch["global_id"]
        accuracies = batch.non_tensor_batch["accuracy"]
        entropies = batch.non_tensor_batch["entropies"]
        high_entropy_token_nums = batch.non_tensor_batch["high_entropy_token_num"]
        difficulty_coefficient = {
            "easy": 1.0,
            "medium": 1.0,
            "hard": 1.0,
        }
        uid_info = defaultdict(lambda: {"acc": [], "entropy": [], "high_entropy_token_num": []})
        difficulty_bucket = defaultdict(list)

        for gid, acc, entropy, tokennum in zip(global_ids, accuracies, entropies, high_entropy_token_nums):
            uid_info[gid]["acc"].append(acc)
            uid_info[gid]["entropy"].append(entropy)
            uid_info[gid]["high_entropy_token_num"].append(tokennum)

        for gid, info in uid_info.items():
            acc_rate = np.mean(info["acc"])
            if not hasattr(self, "bootstrap_acc_dict"):                                                                                                      
                self.bootstrap_acc_dict = {}                                                                                                                 
            self.bootstrap_acc_dict[gid] = float(acc_rate)
            mean_token_num = np.mean(info["high_entropy_token_num"])
            # dynamic sampling here
            # if acc_rate == 1.0 and mean_entropy < self.drop_allpass_entropy_threshold:
            #     self.skip_gid_set.add(gid)
            if acc_rate >= 2 / 3:
                difficulty = "easy"
            elif acc_rate >= 1 / 3:
                difficulty = "medium"
            else:
                difficulty = "hard"

            self.difficulty_dict_train[gid] = difficulty
            difficulty_bucket[difficulty].append(mean_token_num)

        for diff, values in difficulty_bucket.items():
            self.target_high_entropy_token_num_dict[diff] = round(difficulty_coefficient[diff] * np.mean(values))

        # Update alpha_entropy per bucket using deviation from target
        for diff, values in difficulty_bucket.items():
            if not values:
                continue
            mean_gen = float(np.mean(values))
            target = float(self.target_high_entropy_token_num_dict.get(diff, mean_gen))
            if diff in ("easy", "medium"):
                delta = (mean_gen - target)  # too many -> increase penalty
            else:  # hard
                delta = (target - mean_gen)  # not enough -> increase encouragement
            new_alpha = self.alpha_entropy_dict[diff] + self.alpha_entropy_lr * delta
            new_alpha = float(np.clip(new_alpha, self.alpha_entropy_min, self.alpha_entropy_max))
            self.alpha_entropy_dict[diff] = new_alpha
        print(f"[Train] Updated difficulty_dict, size={len(self.difficulty_dict_train)}")


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        # main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        safe_print(f"🚀🚀 Starting training loop. Total steps: {self.training_steps}")
        val_metrics: Optional[dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        # main_tqdm.update(self.global_step)
        if self.global_step > 0: safe_print(f"🚀🚀 Resuming training from step {self.global_step}")

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            safe_print(f"Step {self.global_step}: Performing pre-training validation...")
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            # 打印验证结果摘要
            val_metrics_str = ", ".join(f"{k}: {v:.3g}" for k, v in val_metrics.items())
            safe_print(f"Step {self.global_step}: Pre-train validation done. \n Val Metrics: {val_metrics_str}")
            if self.config.trainer.val_only:
                safe_print("Configured for validation only. Exiting.")
                return
        # #--------------------------New-----------------------------------------#
        # if self.config.data.bootstrap and self.global_step == 0:
        #     self.bootstrap()
        #     self.train_dataloader = create_filtered_train_dataloader(
        #     config=self.config.data,
        #     base_dataset=self.base_dataset,
        #     skip_gid_set=self.skip_gid_set
        #     )
        #     self.data_iterator = iter(self.train_dataloader)
        #     self.logger.log(
        #         data={
        #             "epoch/id": self.epoch_id,
        #             "epoch/skip_gid_count": len(self.skip_gid_set),
        #             "epoch/iter_num":len(self.train_dataloader) 
        #         },
        #         step=self.global_step,
        #     )
        # #--------------------------New-----------------------------------------#

        self.data_iterator = iter(self.train_dataloader)
        train_start_time = time.time()
        while self.global_step < self.training_steps:
            self.global_step += 1
            safe_print(f"**************************************")
            safe_print(f"[{self.global_step}/{self.training_steps}] Step started...")

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    if batch is None:
                        self.actor_rollout_ref_wg.release_rollout_engine()
                        self.epoch_id += 1
                        self.train_dataloader = create_filtered_train_dataloader(
                        config=self.config.data,
                        base_dataset=self.base_dataset,
                        skip_gid_set=self.skip_gid_set
                        )
                        self.data_iterator = iter(self.train_dataloader)
                        continue
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                #--------------------------New-----------------------------------------#
                correctness = batch.non_tensor_batch.get("correctness", None)
                if correctness is not None:
                    batch.non_tensor_batch['accuracy'] = [1.0 if c == 1.0 else 0.0 for c in correctness]
                else:
                    batch.non_tensor_batch['accuracy'] = self.compute_batch_accuracy(batch)
                # use training batch to determine the high_entropy_token_threshold
                global_ids = batch.non_tensor_batch["global_id"]
                batch_thresholds = batch.non_tensor_batch["entropy_percentile_thresholds"]
                # Use robust statistic then EMA to stabilize threshold
                batch_p95 = float(np.median(batch_thresholds))
                if not hasattr(self, "_ema_entropy_threshold"):
                    self._ema_entropy_threshold = batch_p95
                else:
                    self._ema_entropy_threshold = 0.99 * self._ema_entropy_threshold + 0.01 * batch_p95
                self.high_entropy_token_threshold = float(self._ema_entropy_threshold)
                self.update_difficulty_and_skip_gid_set(batch)
                batch.non_tensor_batch["difficulty"] = [self.difficulty_dict_train[gid] for gid in global_ids]
                batch.non_tensor_batch["bootstrap_acc"] = [self.bootstrap_acc_dict.get(gid, 0.0) for gid in global_ids] 
                batch.non_tensor_batch["target_high_entropy_token_num"] = [self.target_high_entropy_token_num_dict[self.difficulty_dict_train[gid]] for gid in global_ids]
                batch.non_tensor_batch["alpha_entropy"] = [self.alpha_entropy_dict[self.difficulty_dict_train[gid]] for gid in global_ids]

                # --- Dual-ascent lambda for reasoning cost (soft) ---
                # Initialize per-difficulty lambda and cost budget if not exist
                if not hasattr(self, "lambda_entropy_dict"):
                    self.lambda_entropy_dict = {"easy": 0.0, "medium": 0.0, "hard": 0.0}
                if not hasattr(self, "lambda_entropy_lr"):
                    # small step to avoid oscillation
                    self.lambda_entropy_lr = 0.01
                if not hasattr(self, "lambda_entropy_max"):
                    self.lambda_entropy_max = 10.0
                if not hasattr(self, "cost_budget_dict"):
                    # reuse target_high_entropy_token_num as budget proxy when soft cost is unavailable
                    self.cost_budget_dict = {
                        "easy": float(self.target_high_entropy_token_num_dict.get("easy", 1.0)),
                        "medium": float(self.target_high_entropy_token_num_dict.get("medium", 1.0)),
                        "hard": float(self.target_high_entropy_token_num_dict.get("hard", 1.0)),
                    }

                # Compute per-difficulty mean soft cost if available
                soft_costs = batch.non_tensor_batch.get("reasoning_soft_cost", None)
                if soft_costs is not None:
                    gid2soft = {gid: float(sc) for gid, sc in zip(global_ids, soft_costs)}
                    bucket_vals = {"easy": [], "medium": [], "hard": []}
                    for gid in global_ids:
                        diff = self.difficulty_dict_train[gid]
                        if diff in bucket_vals:
                            bucket_vals[diff].append(gid2soft.get(gid, 0.0))
                    # update lambda via dual ascent per bucket
                    for diff, vals in bucket_vals.items():
                        if len(vals) < 8:
                            continue  # avoid noise when few samples
                        mean_cost = float(np.mean(vals))
                        target_B = float(self.cost_budget_dict.get(diff, mean_cost))
                        err = mean_cost - target_B
                        new_lam = float(self.lambda_entropy_dict.get(diff, 0.0)) + self.lambda_entropy_lr * err
                        new_lam = float(np.clip(new_lam, 0.0, self.lambda_entropy_max))
                        self.lambda_entropy_dict[diff] = new_lam
                # inject per-sample lambda to batch
                batch.non_tensor_batch["lambda_entropy"] = [self.lambda_entropy_dict[self.difficulty_dict_train[gid]] for gid in global_ids]
                #--------------------------New-----------------------------------------#

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # compute reward
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)
                
                #--------------------------New-----------------------------------------#
                # log per-bucket accuracy/length on this batch
                try:
                    difficulty_list = batch.non_tensor_batch.get("difficulty", None)
                    acc_list = batch.non_tensor_batch.get("accuracy", None)
                    if difficulty_list is not None and acc_list is not None:
                        resp_len = batch.batch["response_mask"].sum(dim=-1).cpu().tolist()
                        per_bucket = {"easy":[], "medium":[], "hard":[]}
                        per_bucket_len = {"easy":[], "medium":[], "hard":[]}
                        for d,a,L in zip(difficulty_list, acc_list, resp_len):
                            key = str(d)
                            if key in per_bucket:
                                per_bucket[key].append(float(a))
                                per_bucket_len[key].append(int(L))
                        for k in per_bucket:
                            if len(per_bucket[k])>0:
                                metrics[f"train/{k}/acc_mean"] = float(np.mean(per_bucket[k]))
                                metrics[f"train/{k}/len_mean"] = float(np.mean(per_bucket_len[k]))
                        # log progress deltas every 5 steps based on bootstrap baseline
                        if hasattr(self, "bootstrap_baseline") and (self.global_step % 10 == 0):
                            for k in ("easy","medium","hard"):
                                if (f"train/{k}/acc_mean" in metrics) and (k in self.bootstrap_baseline):
                                    metrics[f"progress/{k}/dacc"] = metrics[f"train/{k}/acc_mean"] - self.bootstrap_baseline[k]["acc_mean"]
                                    metrics[f"progress/{k}/dlen"] = metrics[f"train/{k}/len_mean"] - self.bootstrap_baseline[k]["len_mean"]
                except Exception:
                    pass
                #--------------------------New-----------------------------------------#

                # recompute old_log_probs
                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)

                # compute ref_log_probs
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                # compute values
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        safe_print(f"[{self.global_step}/{self.training_steps}] Running validation...")
                        val_metrics = self._validate()

                    metrics.update(val_metrics)
                    val_metrics_str = ", ".join(f"{k}: {v:.3g}" for k, v in val_metrics.items())
                    safe_print(f"[{self.global_step}/{self.training_steps}] Val Metrics: {val_metrics_str}")
                    safe_print(f"[{self.global_step}/{self.training_steps}] Validation done.")

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        safe_print(f"[{self.global_step}/{self.training_steps}] Saving checkpoint...")
                        self._save_checkpoint()
                        safe_print(f"[{self.global_step}/{self.training_steps}] Checkpoint saved.")

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)

            # [新增] 计算剩余时间 (ETA)
            elapsed_time = time.time() - train_start_time
            # 防止除零错误，计算平均每步耗时
            avg_time_per_step = elapsed_time / self.global_step if self.global_step > 0 else 0
            remaining_steps = self.training_steps - self.global_step
            eta_seconds = int(avg_time_per_step * remaining_steps)
            # 格式化为 HH:MM:SS
            eta_str = str(datetime.timedelta(seconds=eta_seconds))

            # self._mox_tensorboard2obs() # NOTE moxing tensorboard to obs
            metrics_str = ", ".join(f"{k}: {v:.3g}" for k, v in metrics.items())
            safe_print(f"✅ Step {self.global_step}/{self.training_steps} finished | ETA: {eta_str} \n"
                  f"Train Metrics: {metrics_str}")
            safe_print(f"**************************************")
            # main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                safe_print("Running final validation...")
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            safe_print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            safe_print("Saving final checkpoint...")
            self._save_checkpoint()
            safe_print("Done.")