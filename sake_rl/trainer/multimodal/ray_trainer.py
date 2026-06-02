import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pprint import pprint
from typing import Type

import numpy as np
import pandas as pd
import tqdm
from omegaconf import OmegaConf, open_dict
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import compute_timing_metrics, reduce_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    Role,
    _timer,
    apply_kl_penalty,
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.seqlen_balancing import (
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)

from sake_rl.monkey_patch.monkey_patch import create_colocated_worker_cls_patch
from sake_rl.trainer.multimodal.core_algos import compute_grpo_outcome_advantage
from sake_rl.utils.dataset.mm_rl_dataset import RLHFDataset, collate_fn

WorkerType = Type[Worker]
import torch


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True, tokenizer=None):
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    # Add 'Search_Ratio' and 'Accuracy' metrics for Twitter-GMNER
    total_count = sequence_score.size(0)
    responses_after_first_user_prompt = batch.batch['responses']
    assert total_count == responses_after_first_user_prompt.shape[0], "B*R != Total Num of Rollout Responses"
    
    # Collect F1 metrics for Twitter-GMNER
    f1_scores = []
    precisions = []
    recalls = []
    tps, fps, fns = [], [], []
    
    # MultiEM metrics
    mner_f1_scores, mner_precisions, mner_recalls = [], [], []
    mner_tps, mner_fps, mner_fns = [], [], []
    eeg_f1_scores, eeg_precisions, eeg_recalls = [], [], []
    eeg_tps, eeg_fps, eeg_fns = [], [], []
    gmner_f1_scores, gmner_precisions, gmner_recalls = [], [], []
    gmner_tps, gmner_fps, gmner_fns = [], [], []
    
    # Detect reward mode
    reward_mode = 'EM'
    if 'extra_info' in batch.non_tensor_batch and len(batch.non_tensor_batch['extra_info']) > 0:
        first_extra_info = batch.non_tensor_batch['extra_info'][0]
        if hasattr(first_extra_info, 'item'):
            first_extra_info = first_extra_info.item()
        if isinstance(first_extra_info, dict) and 'reward_mode' in first_extra_info:
            reward_mode = first_extra_info['reward_mode']
    
    for idx, response in enumerate(responses_after_first_user_prompt):
        _resp_length = response.size(0)
        if 'multi_turn_response_mask' in batch.batch:
            _resp_mask = batch.batch['multi_turn_response_mask'][idx][-_resp_length:]
            response, response_non_assistant = response[_resp_mask == 1], response[_resp_mask < 0.1]
        response_non_assistant = tokenizer.decode(response_non_assistant)
        
        # Extract F1 metrics if available (from extra_info)
        if 'extra_info' in batch.non_tensor_batch:
            try:
                extra_info_list = batch.non_tensor_batch['extra_info']
                # Handle both list and numpy array cases
                if hasattr(extra_info_list, '__len__') and idx < len(extra_info_list):
                    extra_info = extra_info_list[idx]
                    # Handle numpy array case: convert to dict if it's a numpy object
                    if hasattr(extra_info, 'item'):  # numpy scalar
                        extra_info = extra_info.item()
                    if isinstance(extra_info, dict):
                        # Common metrics (for backward compatibility)
                        if 'f1_score' in extra_info:
                            f1_scores.append(extra_info['f1_score'])
                        if 'precision' in extra_info:
                            precisions.append(extra_info['precision'])
                        if 'recall' in extra_info:
                            recalls.append(extra_info['recall'])
                        if 'tp' in extra_info:
                            tps.append(extra_info['tp'])
                        if 'fp' in extra_info:
                            fps.append(extra_info['fp'])
                        if 'fn' in extra_info:
                            fns.append(extra_info['fn'])
                        
                        # MultiEM specific metrics
                        if reward_mode == 'MultiEM':
                            if 'mner_f1_score' in extra_info:
                                mner_f1_scores.append(extra_info['mner_f1_score'])
                            if 'mner_precision' in extra_info:
                                mner_precisions.append(extra_info['mner_precision'])
                            if 'mner_recall' in extra_info:
                                mner_recalls.append(extra_info['mner_recall'])
                            if 'mner_tp' in extra_info:
                                mner_tps.append(extra_info['mner_tp'])
                            if 'mner_fp' in extra_info:
                                mner_fps.append(extra_info['mner_fp'])
                            if 'mner_fn' in extra_info:
                                mner_fns.append(extra_info['mner_fn'])
                            
                            if 'eeg_f1_score' in extra_info:
                                eeg_f1_scores.append(extra_info['eeg_f1_score'])
                            if 'eeg_precision' in extra_info:
                                eeg_precisions.append(extra_info['eeg_precision'])
                            if 'eeg_recall' in extra_info:
                                eeg_recalls.append(extra_info['eeg_recall'])
                            if 'eeg_tp' in extra_info:
                                eeg_tps.append(extra_info['eeg_tp'])
                            if 'eeg_fp' in extra_info:
                                eeg_fps.append(extra_info['eeg_fp'])
                            if 'eeg_fn' in extra_info:
                                eeg_fns.append(extra_info['eeg_fn'])
                            
                            if 'gmner_f1_score' in extra_info:
                                gmner_f1_scores.append(extra_info['gmner_f1_score'])
                            if 'gmner_precision' in extra_info:
                                gmner_precisions.append(extra_info['gmner_precision'])
                            if 'gmner_recall' in extra_info:
                                gmner_recalls.append(extra_info['gmner_recall'])
                            if 'gmner_tp' in extra_info:
                                gmner_tps.append(extra_info['gmner_tp'])
                            if 'gmner_fp' in extra_info:
                                gmner_fps.append(extra_info['gmner_fp'])
                            if 'gmner_fn' in extra_info:
                                gmner_fns.append(extra_info['gmner_fn'])
            except (IndexError, TypeError, KeyError) as e:
                # Silently skip if extra_info is not accessible
                pass

    fp = 0.1
    if 'extra_info' in batch.non_tensor_batch and 'format_penalty' in batch.non_tensor_batch['extra_info'][0]:
        fp = batch.non_tensor_batch['extra_info'][0]['format_penalty']
    correct_threshold = fp + 1e-4
    count_correct = torch.sum(sequence_score > correct_threshold).item()
    answer_acc = count_correct / total_count

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    # update response_length 
    if 'multi_turn_response_mask' in batch.batch:
        response_length = batch.batch['multi_turn_response_mask'].sum(dim=1).float()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # Twitter-GMNER F1 metrics
        **(
            {
                'critic/twitter_gmner/f1_score_mean': np.mean(f1_scores),
                'critic/twitter_gmner/precision_mean': np.mean(precisions),
                'critic/twitter_gmner/recall_mean': np.mean(recalls),
                'critic/twitter_gmner/tp_sum': np.sum(tps),
                'critic/twitter_gmner/fp_sum': np.sum(fps),
                'critic/twitter_gmner/fn_sum': np.sum(fns),
            }
            if len(f1_scores) > 0
            else {}
        ),
        # MultiEM specific metrics
        **(
            {
                'critic/twitter_gmner/mner_f1_score_mean': np.mean(mner_f1_scores),
                'critic/twitter_gmner/mner_precision_mean': np.mean(mner_precisions),
                'critic/twitter_gmner/mner_recall_mean': np.mean(mner_recalls),
                'critic/twitter_gmner/mner_tp_sum': np.sum(mner_tps),
                'critic/twitter_gmner/mner_fp_sum': np.sum(mner_fps),
                'critic/twitter_gmner/mner_fn_sum': np.sum(mner_fns),
                'critic/twitter_gmner/eeg_f1_score_mean': np.mean(eeg_f1_scores),
                'critic/twitter_gmner/eeg_precision_mean': np.mean(eeg_precisions),
                'critic/twitter_gmner/eeg_recall_mean': np.mean(eeg_recalls),
                'critic/twitter_gmner/eeg_tp_sum': np.sum(eeg_tps),
                'critic/twitter_gmner/eeg_fp_sum': np.sum(eeg_fps),
                'critic/twitter_gmner/eeg_fn_sum': np.sum(eeg_fns),
                'critic/twitter_gmner/gmner_f1_score_mean': np.mean(gmner_f1_scores),
                'critic/twitter_gmner/gmner_precision_mean': np.mean(gmner_precisions),
                'critic/twitter_gmner/gmner_recall_mean': np.mean(gmner_recalls),
                'critic/twitter_gmner/gmner_tp_sum': np.sum(gmner_tps),
                'critic/twitter_gmner/gmner_fp_sum': np.sum(gmner_fps),
                'critic/twitter_gmner/gmner_fn_sum': np.sum(gmner_fns),
            }
            if reward_mode == 'MultiEM' and len(mner_f1_scores) > 0
            else {}
        ),
        # score
        'critic/score/mean': torch.mean(sequence_score).detach().item(),
        'critic/score/max': torch.max(sequence_score).detach().item(),
        'critic/score/min': torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean': torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max': torch.max(sequence_reward).detach().item(),
        'critic/rewards/min': torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean': torch.mean(valid_adv).detach().item(),
        'critic/advantages/max': torch.max(valid_adv).detach().item(),
        'critic/advantages/min': torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean': torch.mean(valid_returns).detach().item(),
        'critic/returns/max': torch.max(valid_returns).detach().item(),
        'critic/returns/min': torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                'critic/values/mean': torch.mean(valid_values).detach().item(),
                'critic/values/max': torch.max(valid_values).detach().item(),
                'critic/values/min': torch.min(valid_values).detach().item(),
                # vf explained var
                'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        'response_length/mean': torch.mean(response_length).detach().item(),
        'response_length/max': torch.max(response_length).detach().item(),
        'response_length/min': torch.min(response_length).detach().item(),
        'response_length/clip_ratio': torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # prompt length
        'prompt_length/mean': torch.mean(prompt_length).detach().item(),
        'prompt_length/max': torch.max(prompt_length).detach().item(),
        'prompt_length/min': torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio': torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_advantage(
    data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, grpo_normalize=True
):  
    # normalize controls whether grpo divides group std
    if adv_estimator == AdvantageEstimator.GRPO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            eos_mask=response_mask,
            index=index,
            grpo_normalize=grpo_normalize,
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
    ):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert (
                    config.algorithm.kl_ctrl.horizon > 0
                ), f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(
                    init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                    target_kl=config.algorithm.kl_ctrl.target_kl,
                    horizon=config.algorithm.kl_ctrl.horizon,
                )
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.0)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert (
            real_train_batch_size % n_gpus == 0
        ), f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(
                    f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                    f"'{name}.micro_batch_size_per_gpu'."
                )

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(
                    f"[{name}] You have set both '{name}.micro_batch_size' AND "
                    f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                    f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated)."
                )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.ref",
            )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if (
                config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1
                or config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1
            ):
                assert (
                    config.actor_rollout_ref.model.use_remove_padding
                ), "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert (
                    config.critic.model.use_remove_padding
                ), "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(
            parquet_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            image_key=self.config.data.get('image_key', 'images'),
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation='error',
            user_prompt_round_1=self.config.data.get('user_prompt_round_1', None),
        )
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            num_workers=self.config.data.num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        self.val_dataset = RLHFDataset(
            parquet_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            image_key=self.config.data.get('image_key', 'images'),
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation='error',
            user_prompt_round_1=self.config.data.get('user_prompt_round_1', None),
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        assert (
            len(self.val_dataloader) == 1
        ), "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations_to_wandb(self, inputs, outputs, scores, reward_models=[], image_urls=[]):
        """Log a table of validation samples to wandb"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and 'wandb' not in self.config.trainer.logger:
            print(
                'WARNING: `val_generations_to_log_to_wandb` is set to a positive value, but no wandb logger is found. '
            )
            return

        import numpy as np
        import wandb

        also_log_ground_truth_and_image_urls = len(reward_models) > 0 and len(image_urls) > 0

        # Create tuples of (input, output, score, *reward_models, *image_urls) and sort by input text
        if also_log_ground_truth_and_image_urls:
            samples = list(zip(inputs, outputs, scores, reward_models, image_urls))
        else:
            samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Initialize DataFrame columns
        if also_log_ground_truth_and_image_urls:
            columns = ["step", "input_text", "output_text", "score", "reward_model", "image_url"]
        else:
            columns = ["step", "input_text", "output_text", "score"]

        if not hasattr(self, 'validation_table'):
            # Initialize the table on first call
            self.validation_table = pd.DataFrame(columns=columns)

        # Create a DataFrame for the current step's validation
        row_data = []
        for sample in samples:
            if also_log_ground_truth_and_image_urls:
                reward_model = sample[3]
                if 'candidate_answers' not in reward_model:
                    reward_model['candidate_answers'] = '[]'
                row = {
                    "step": self.global_steps,
                    "input_text": sample[0],
                    "output_text": sample[1],
                    "score": sample[2],
                    "reward_model": reward_model,  # NOTE: strict sequence
                    "image_url": sample[4],
                }
            else:
                row = {"step": self.global_steps, "input_text": sample[0], "output_text": sample[1], "score": sample[2]}
            row_data.append(row)

        # Convert to DataFrame and append to existing data
        new_df = pd.DataFrame(row_data)
        # Updated table for future logging
        self.validation_table = pd.concat([self.validation_table, new_df], ignore_index=True)
        
        # NOTE: We mannualy save eval results for offline 'eval_only' Mode
        if self.config.trainer.get('val_only', False) and self.config.trainer.val_only_save_dir is not None:
            os.makedirs(self.config.trainer.val_only_save_dir, exist_ok=True)
            save_path = os.path.join(
                self.config.trainer.val_only_save_dir, f"val_result_{len(self.validation_table)}.json"
            )
            self.validation_table.to_json(save_path, orient="records", indent=2)
            print(f'validation generation saved to local: {save_path}')

        # Update reference and log
        wandb.log({"val/generations": wandb.Table(dataframe=self.validation_table)}, step=self.global_steps)
        print('validation generation saved to wandb table')

    def _validate(self):

        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Validate Starts ..."
        )

        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        # NOTE: we collect ground truth and image_url also
        sample_reward_models = []
        sample_image_url = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            # Store ground truth
            if 'reward_model' in test_batch.non_tensor_batch:
                sample_reward_models.extend(list(test_batch.non_tensor_batch['reward_model']))
            # Store image_url
            if 'image_urls' in test_batch.non_tensor_batch:
                sample_image_url.extend(list(test_batch.non_tensor_batch['image_urls']))

            if 'multi_modal_data' in test_batch.non_tensor_batch.keys():
                if 'image_urls' in test_batch.non_tensor_batch:
                    test_gen_batch = test_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'image_urls'],
                    )
                else:
                    test_gen_batch = test_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data'],
                    )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            if 'extra_info' not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch['extra_info'] = [{} for _ in range(len(test_batch.batch))]
            for item_id in range(len(test_batch.batch)):
                if 'search_penalty' in self.config.trainer:
                    test_batch.non_tensor_batch['extra_info'][item_id][
                        'search_penalty'
                    ] = self.config.trainer.search_penalty
                if 'format_penalty' in self.config.trainer:
                    test_batch.non_tensor_batch['extra_info'][item_id][
                        'format_penalty'
                    ] = self.config.trainer.format_penalty
                if 'reward_mode' in self.config.trainer:
                    assert self.config.trainer.reward_mode in [
                        'EM',
                        'MultiEM',
                    ], f'reward mode {self.config.trainer.reward_mode} not recognized, please use EM or MultiEM'
                    test_batch.non_tensor_batch['extra_info'][item_id]['reward_mode'] = self.config.trainer.reward_mode
                    # Set MultiEM weights if using MultiEM
                    if self.config.trainer.reward_mode == 'MultiEM':
                        test_batch.non_tensor_batch['extra_info'][item_id]['lambda_1'] = getattr(self.config.trainer, 'lambda_1', 0.25)
                        test_batch.non_tensor_batch['extra_info'][item_id]['lambda_2'] = getattr(self.config.trainer, 'lambda_2', 0.25)
                        test_batch.non_tensor_batch['extra_info'][item_id]['lambda_3'] = getattr(self.config.trainer, 'lambda_3', 0.5)
                if 'use_search_count_penalty' in self.config.trainer:
                    use_search_count_penalty = self.config.trainer.use_search_count_penalty
                    test_batch.non_tensor_batch['extra_info'][item_id][
                        'use_search_count_penalty'
                    ] = use_search_count_penalty
            test_batch.non_tensor_batch['extra_info'] = np.array(test_batch.non_tensor_batch['extra_info'])
            reward_tensor = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
        
        self._maybe_log_val_generations_to_wandb(
            inputs=sample_inputs,
            outputs=sample_outputs,
            scores=sample_scores,
            reward_models=sample_reward_models,
            image_urls=sample_image_url,
        )

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        # Collect F1 metrics for validation (similar to training)
        reward_mode = getattr(self.config.trainer, 'reward_mode', 'EM')
        val_f1_scores, val_precisions, val_recalls = [], [], []
        val_tps, val_fps, val_fns = [], [], []
        val_mner_f1_scores, val_mner_precisions, val_mner_recalls = [], [], []
        val_mner_tps, val_mner_fps, val_mner_fns = [], [], []
        val_eeg_f1_scores, val_eeg_precisions, val_eeg_recalls = [], [], []
        val_eeg_tps, val_eeg_fps, val_eeg_fns = [], [], []
        val_gmner_f1_scores, val_gmner_precisions, val_gmner_recalls = [], [], []
        val_gmner_tps, val_gmner_fps, val_gmner_fns = [], [], []
        
        # Extract metrics from test_batch extra_info
        if 'extra_info' in test_batch.non_tensor_batch:
            for idx in range(len(test_batch.batch)):
                try:
                    extra_info_list = test_batch.non_tensor_batch['extra_info']
                    if hasattr(extra_info_list, '__len__') and idx < len(extra_info_list):
                        extra_info = extra_info_list[idx]
                        if hasattr(extra_info, 'item'):
                            extra_info = extra_info.item()
                        if isinstance(extra_info, dict):
                            # Common metrics
                            if 'f1_score' in extra_info:
                                val_f1_scores.append(extra_info['f1_score'])
                            if 'precision' in extra_info:
                                val_precisions.append(extra_info['precision'])
                            if 'recall' in extra_info:
                                val_recalls.append(extra_info['recall'])
                            if 'tp' in extra_info:
                                val_tps.append(extra_info['tp'])
                            if 'fp' in extra_info:
                                val_fps.append(extra_info['fp'])
                            if 'fn' in extra_info:
                                val_fns.append(extra_info['fn'])
                            
                            # MultiEM specific metrics
                            if reward_mode == 'MultiEM':
                                if 'mner_f1_score' in extra_info:
                                    val_mner_f1_scores.append(extra_info['mner_f1_score'])
                                if 'mner_precision' in extra_info:
                                    val_mner_precisions.append(extra_info['mner_precision'])
                                if 'mner_recall' in extra_info:
                                    val_mner_recalls.append(extra_info['mner_recall'])
                                if 'mner_tp' in extra_info:
                                    val_mner_tps.append(extra_info['mner_tp'])
                                if 'mner_fp' in extra_info:
                                    val_mner_fps.append(extra_info['mner_fp'])
                                if 'mner_fn' in extra_info:
                                    val_mner_fns.append(extra_info['mner_fn'])
                                
                                if 'eeg_f1_score' in extra_info:
                                    val_eeg_f1_scores.append(extra_info['eeg_f1_score'])
                                if 'eeg_precision' in extra_info:
                                    val_eeg_precisions.append(extra_info['eeg_precision'])
                                if 'eeg_recall' in extra_info:
                                    val_eeg_recalls.append(extra_info['eeg_recall'])
                                if 'eeg_tp' in extra_info:
                                    val_eeg_tps.append(extra_info['eeg_tp'])
                                if 'eeg_fp' in extra_info:
                                    val_eeg_fps.append(extra_info['eeg_fp'])
                                if 'eeg_fn' in extra_info:
                                    val_eeg_fns.append(extra_info['eeg_fn'])
                                
                                if 'gmner_f1_score' in extra_info:
                                    val_gmner_f1_scores.append(extra_info['gmner_f1_score'])
                                if 'gmner_precision' in extra_info:
                                    val_gmner_precisions.append(extra_info['gmner_precision'])
                                if 'gmner_recall' in extra_info:
                                    val_gmner_recalls.append(extra_info['gmner_recall'])
                                if 'gmner_tp' in extra_info:
                                    val_gmner_tps.append(extra_info['gmner_tp'])
                                if 'gmner_fp' in extra_info:
                                    val_gmner_fps.append(extra_info['gmner_fp'])
                                if 'gmner_fn' in extra_info:
                                    val_gmner_fns.append(extra_info['gmner_fn'])
                except (IndexError, TypeError, KeyError):
                    pass
        
        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/{data_source}/reward'] = np.mean(rewards)
            fp = 0.1
            if 'format_penalty' in self.config.trainer:  # adjust for other format reward
                fp = self.config.trainer.format_penalty
            correct_threshold = fp + 1e-4
            correct_cnt = sum(1 for x in rewards if x > correct_threshold)
            metric_dict[f'val/{data_source}/score'] = correct_cnt / len(rewards)
            
            # Add F1 metrics for validation
            if len(val_f1_scores) > 0:
                metric_dict[f'val/{data_source}/f1_score_mean'] = np.mean(val_f1_scores)
                metric_dict[f'val/{data_source}/precision_mean'] = np.mean(val_precisions)
                metric_dict[f'val/{data_source}/recall_mean'] = np.mean(val_recalls)
                metric_dict[f'val/{data_source}/tp_sum'] = np.sum(val_tps)
                metric_dict[f'val/{data_source}/fp_sum'] = np.sum(val_fps)
                metric_dict[f'val/{data_source}/fn_sum'] = np.sum(val_fns)
            
            # Add MultiEM specific metrics for validation
            if reward_mode == 'MultiEM' and len(val_mner_f1_scores) > 0:
                metric_dict[f'val/{data_source}/mner_f1_score_mean'] = np.mean(val_mner_f1_scores)
                metric_dict[f'val/{data_source}/mner_precision_mean'] = np.mean(val_mner_precisions)
                metric_dict[f'val/{data_source}/mner_recall_mean'] = np.mean(val_mner_recalls)
                metric_dict[f'val/{data_source}/mner_tp_sum'] = np.sum(val_mner_tps)
                metric_dict[f'val/{data_source}/mner_fp_sum'] = np.sum(val_mner_fps)
                metric_dict[f'val/{data_source}/mner_fn_sum'] = np.sum(val_mner_fns)
                
                metric_dict[f'val/{data_source}/eeg_f1_score_mean'] = np.mean(val_eeg_f1_scores)
                metric_dict[f'val/{data_source}/eeg_precision_mean'] = np.mean(val_eeg_precisions)
                metric_dict[f'val/{data_source}/eeg_recall_mean'] = np.mean(val_eeg_recalls)
                metric_dict[f'val/{data_source}/eeg_tp_sum'] = np.sum(val_eeg_tps)
                metric_dict[f'val/{data_source}/eeg_fp_sum'] = np.sum(val_eeg_fps)
                metric_dict[f'val/{data_source}/eeg_fn_sum'] = np.sum(val_eeg_fns)
                
                metric_dict[f'val/{data_source}/gmner_f1_score_mean'] = np.mean(val_gmner_f1_scores)
                metric_dict[f'val/{data_source}/gmner_precision_mean'] = np.mean(val_gmner_precisions)
                metric_dict[f'val/{data_source}/gmner_recall_mean'] = np.mean(val_gmner_recalls)
                metric_dict[f'val/{data_source}/gmner_tp_sum'] = np.sum(val_gmner_tps)
                metric_dict[f'val/{data_source}/gmner_fp_sum'] = np.sum(val_gmner_fps)
                metric_dict[f'val/{data_source}/gmner_fn_sum'] = np.sum(val_gmner_fns)
            
            responses_after_first_user_prompt = test_batch.batch['responses']
            for idx, response in enumerate(responses_after_first_user_prompt):
                _resp_length = response.size(0)
                if 'multi_turn_response_mask' in test_batch.batch:
                    _resp_mask = test_batch.batch['multi_turn_response_mask'][idx][-_resp_length:]
                    response, response_non_assistant = response[_resp_mask == 1], response[_resp_mask < 0.1]
                response_non_assistant = self.tokenizer.decode(response_non_assistant)
            metric_dict[f'val/{data_source}/correct_threshold'] = correct_threshold
            metric_dict[f'val/{data_source}/rewards_len'] = len(rewards)
            metric_dict[f'val/{data_source}/responses_after_first_user_prompt_len'] = len(
                responses_after_first_user_prompt
            )

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Validate Ends ...")

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role='actor_rollout',
            )
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role='ref'
            )
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls_patch(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f'global_step_{self.global_steps}'
        )
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        )
        remove_previous_ckpt_in_save = self.config.trainer.get('remove_previous_ckpt_in_save', False)
        if remove_previous_ckpt_in_save:
            print(
                'Warning: remove_previous_ckpt_in_save is deprecated, set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead'
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get('max_actor_ckpt_to_keep', None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get('max_critic_ckpt_to_keep', None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, 'latest_checkpointed_iteration.txt'
        )
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
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

    def _convert_to_serializable(self, obj):
        """Convert numpy types and other non-serializable objects to Python native types."""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist() if obj.numel() > 1 else obj.item()
        elif hasattr(obj, 'item'):  # Handle other numpy scalar types
            return obj.item()
        else:
            return obj

    def _save_metrics_to_json(self, metrics_history: list, step: int):
        """Save metrics history to local JSON file."""
        metrics_log_path = self.config.trainer.get('metrics_log_path', None)
        if metrics_log_path is None:
            # Default path
            metrics_log_path = os.path.join(
                self.config.trainer.default_local_dir,
                'metrics_history.json'
            )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(metrics_log_path), exist_ok=True)
        
        try:
            # Convert all metrics to JSON-serializable types
            serializable_history = self._convert_to_serializable(metrics_history)
            with open(metrics_log_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_history, f, indent=2, ensure_ascii=False)
            if step % 10 == 0:  # Only print every 10 steps to reduce log spam
                print(f"[Metrics] Saved {len(metrics_history)} steps to {metrics_log_path}")
        except Exception as e:
            print(f"[Metrics] Failed to save metrics to {metrics_log_path}: {e}")
            import traceback
            traceback.print_exc()

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        
        # Initialize metrics history for local JSON saving
        metrics_history = []

        # load checkpoint before doing anything
        self._load_checkpoint()
        
        # Try to load existing metrics history if resuming
        metrics_log_path = self.config.trainer.get('metrics_log_path', None)
        if metrics_log_path is None:
            metrics_log_path = os.path.join(
                self.config.trainer.default_local_dir,
                'metrics_history.json'
            )
        if os.path.exists(metrics_log_path) and self.global_steps > 0:
            try:
                with open(metrics_log_path, 'r', encoding='utf-8') as f:
                    metrics_history = json.load(f)
                print(f"[Metrics] Loaded {len(metrics_history)} existing metrics from {metrics_log_path}")
            except Exception as e:
                print(f"[Metrics] Failed to load existing metrics: {e}, starting fresh")

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            
            # Save initial validation metrics to local JSON
            metrics_entry = {
                'step': self.global_steps,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'type': 'validation_initial',
                **val_metrics
            }
            metrics_history.append(metrics_entry)
            self._save_metrics_to_json(metrics_history, self.global_steps)
            
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1
        
        # DAPO variable for rejection sampling continue generation
        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0

        # add tqdm
        progress_bar = tqdm.tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                # multi-round generation for rejections sampling, until a batch of diverse rewards is generated
                # we use new_batch to replace batch in verl's single round generation
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if 'multi_modal_data' in new_batch.non_tensor_batch.keys():
                    if (
                        'image_urls' in new_batch.non_tensor_batch
                    ):  # field 'image_urls' is used for potential search actions
                        gen_batch = new_batch.pop(
                            batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                            non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'image_urls'],
                        )
                    else:
                        gen_batch = new_batch.pop(
                            batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                            non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data'],
                        )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )
                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Rollout Starts ..."
                    )
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        del gen_batch  # FIXME: cause error when "self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX"
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Rollout Ends ..."
                    )

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch['uid'] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with _timer('reward', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)
                        if 'extra_info' not in new_batch.non_tensor_batch:
                            new_batch.non_tensor_batch['extra_info'] = [{} for _ in range(len(new_batch.batch))]
                        for item_id in range(len(new_batch.batch)):
                            if 'search_penalty' in self.config.trainer:
                                step_search_penalty = self.config.trainer.search_penalty
                                if 'search_penalty_warmup_steps' in self.config.trainer:
                                    step_search_penalty = -self.config.trainer.search_penalty
                                    step_search_penalty += min(
                                        (self.global_steps - 1) / self.config.trainer.search_penalty_warmup_steps, 1
                                    ) * (2 * self.config.trainer.search_penalty)
                                new_batch.non_tensor_batch['extra_info'][item_id][
                                    'search_penalty'
                                ] = step_search_penalty
                            if 'format_penalty' in self.config.trainer:
                                new_batch.non_tensor_batch['extra_info'][item_id][
                                    'format_penalty'
                                ] = self.config.trainer.format_penalty
                            if 'reward_mode' in self.config.trainer:
                                assert self.config.trainer.reward_mode in [
                                    'EM',
                                    'MultiEM',
                                ], f'reward mode {self.config.trainer.reward_mode} not recognized, please use EM or MultiEM'
                                new_batch.non_tensor_batch['extra_info'][item_id][
                                    'reward_mode'
                                ] = self.config.trainer.reward_mode
                                # Set MultiEM weights if using MultiEM
                                if self.config.trainer.reward_mode == 'MultiEM':
                                    new_batch.non_tensor_batch['extra_info'][item_id]['lambda_1'] = getattr(self.config.trainer, 'lambda_1', 0.25)
                                    new_batch.non_tensor_batch['extra_info'][item_id]['lambda_2'] = getattr(self.config.trainer, 'lambda_2', 0.25)
                                    new_batch.non_tensor_batch['extra_info'][item_id]['lambda_3'] = getattr(self.config.trainer, 'lambda_3', 0.5)
                            # add search count penalty
                            if 'use_search_count_penalty' in self.config.trainer:
                                use_search_count_penalty = self.config.trainer.use_search_count_penalty
                                new_batch.non_tensor_batch['extra_info'][item_id][
                                    'use_search_count_penalty'
                                ] = use_search_count_penalty
                        new_batch.non_tensor_batch['extra_info'] = np.array(new_batch.non_tensor_batch['extra_info'])

                        # we combine with rule-based rm
                        print(
                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Reward_fn Starts ..."
                        )
                        reward_tensor = self.reward_fn(new_batch)
                        print(
                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Reward_fn Ends ..."
                        )
                        new_batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            new_batch.batch['token_level_rewards'] = new_batch.batch['token_level_scores']

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size, we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if (
                            metric_name == "seq_final_reward"
                        ):  # use batch instead of tensor batch because dtype is float not object
                            # Turn to numpy for easier filtering
                            new_batch.batch["seq_final_reward"] = (
                                new_batch.batch['token_level_scores'].sum(dim=-1).numpy()
                            )
                        elif (
                            metric_name == "seq_reward"
                        ):  # use batch instead of tensor batch because dtype is float not object
                            new_batch.batch["seq_reward"] = new_batch.batch['token_level_scores'].sum(dim=-1).numpy()

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch['uid'], new_batch.batch[metric_name]
                        ):  # use batch instead of tensor batch because dtype is float not object
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std > 0]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch['uid']):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        if batch is None:
                            batch = new_batch
                        else:
                            batch = DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f'{num_prompt_in_batch=} < {prompt_bsz=}')
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f'{num_gen_batches=}. Keep generating...')
                                continue
                            else:
                                raise ValueError(
                                    f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.'
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    # self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Compute log_prob Starts ..."
                    )
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Compute log_prob Ends ..."
                    )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute advantages, executed on the driver process
                        if 'grpo_denormalize' in self.config.trainer and self.config.trainer.grpo_denormalize:
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                grpo_normalize=False,
                            )
                        else:
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                grpo_normalize=True,
                            )
                    
                    # update critic
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Update Actor Starts ..."
                    )
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Update Actor Ends ..."
                    )

                    # implement critic warmup
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Critic Warmup Starts ..."
                    )
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Critic Warmup Ends ..."
                    )

                    # validate
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Validation Starts ..."
                    )
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Global Step: {self.global_steps}] Validation Ends ..."
                    )
                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic, tokenizer=self.tokenizer))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                timing_raw = defaultdict(float)  # clear timing
                metrics["train/num_gen_batches"] = num_gen_batches
                if 'search_penalty' in self.config.trainer:
                    metrics["train/search_penalty"] = step_search_penalty

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                
                # Save metrics to local JSON
                metrics_entry = {
                    'step': self.global_steps,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'type': 'training',
                    **metrics
                }
                metrics_history.append(metrics_entry)
                self._save_metrics_to_json(metrics_history, self.global_steps)

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    # Save final metrics
                    final_entry = {
                        'step': self.global_steps,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'type': 'validation_final',
                        **last_val_metrics
                    }
                    metrics_history.append(final_entry)
                    self._save_metrics_to_json(metrics_history, self.global_steps)
                    progress_bar.close()
                    return
                
                progress_bar.update(1)
                self.global_steps += 1

                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0