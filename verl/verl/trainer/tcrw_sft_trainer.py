"""Representation-aware warm-up trainer for TCRW-OPD.

This module extends the standard verl FSDP SFT trainer with an optional
Teacher-Student hidden-state alignment objective during the *off-policy*
warm-up stage. It is intentionally restricted to the non-Ulysses,
non-remove-padding path for the first experimental version so that the
teacher and student hidden-state tensors have an unambiguous [B, T, H]
layout.

The downstream OPD objective is not changed by this trainer.
"""

import logging
import os
import time

import hydra
import torch
import torch.distributed
import torch.nn.functional as F
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoConfig, AutoModelForCausalLM

from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer, create_sft_dataset
from verl.utils import hf_tokenizer
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import fsdp2_clip_grad_norm_
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.tracking import Tracking

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


class TCRWFSDPSFTTrainer(FSDPSFTTrainer):
    """FSDP SFT trainer with a frozen teacher representation loss."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config", args[0] if args else None)
        if config is None:
            raise ValueError("TCRWFSDPSFTTrainer requires a config")

        self.rep_cfg = config.get("representation_warmup", {})
        self.rep_alpha = float(self.rep_cfg.get("alpha", 1.0))
        self.rep_beta = float(self.rep_cfg.get("beta", 0.3))
        self.rep_loss_type = str(self.rep_cfg.get("loss_type", "cosine")).lower()
        self.rep_response_only = bool(self.rep_cfg.get("response_only", True))
        self.rep_layers = [int(x) for x in self.rep_cfg.get("layers", [-1])]
        self.teacher_model = None

        if config.ulysses_sequence_parallel_size != 1:
            raise ValueError("TCRW v1 currently requires ulysses_sequence_parallel_size=1")
        if bool(config.get("use_remove_padding", False)):
            raise ValueError("TCRW v1 currently requires use_remove_padding=False")
        if not self.rep_cfg.get("teacher_model_path", None):
            raise ValueError("representation_warmup.teacher_model_path must be set")
        if self.rep_loss_type not in {"cosine", "normalized_mse"}:
            raise ValueError("representation_warmup.loss_type must be cosine or normalized_mse")

        super().__init__(*args, **kwargs)
        self._build_frozen_teacher()

    def _build_frozen_teacher(self):
        teacher_path = copy_to_local(src=self.rep_cfg.teacher_model_path, verbose=True)
        trust_remote_code = bool(self.config.model.trust_remote_code)
        teacher_cfg = AutoConfig.from_pretrained(teacher_path, trust_remote_code=trust_remote_code)

        if getattr(teacher_cfg, "hidden_size", None) != getattr(self.model_config, "hidden_size", None):
            raise ValueError(
                "TCRW v1 directly aligns hidden states and therefore requires matching hidden sizes. "
                f"student={getattr(self.model_config, 'hidden_size', None)}, "
                f"teacher={getattr(teacher_cfg, 'hidden_size', None)}"
            )

        dtype_name = str(self.rep_cfg.get("teacher_dtype", "bf16")).lower()
        if dtype_name in {"bf16", "bfloat16"}:
            teacher_dtype = torch.bfloat16
        elif dtype_name in {"fp16", "float16"}:
            teacher_dtype = torch.float16
        else:
            teacher_dtype = torch.float32

        log_gpu_memory_usage("Before TCRW teacher allocation", logger=logger)
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_path,
            config=teacher_cfg,
            torch_dtype=teacher_dtype,
            attn_implementation="flash_attention_2",
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        device = torch.device(self.config.trainer.device, torch.cuda.current_device())
        self.teacher_model.to(device)
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        log_gpu_memory_usage("After TCRW teacher allocation", logger=logger)

        if self.device_mesh.get_rank() == 0:
            print(
                "[TCRW] frozen teacher loaded:\n"
                f"  path={self.rep_cfg.teacher_model_path}\n"
                f"  layers={self.rep_layers}\n"
                f"  loss={self.rep_loss_type}\n"
                f"  alpha={self.rep_alpha}, beta={self.rep_beta}\n"
                f"  response_only={self.rep_response_only}"
            )

    @staticmethod
    def _resolve_layer_index(index: int, hidden_states):
        n = len(hidden_states)
        resolved = index if index >= 0 else n + index
        if resolved < 0 or resolved >= n:
            raise IndexError(f"hidden-state layer index {index} resolves to {resolved}, valid range is [0,{n - 1}]")
        return resolved

    def _representation_loss(self, student_hidden, teacher_hidden, token_mask):
        if len(student_hidden) != len(teacher_hidden):
            raise ValueError(
                f"Student/teacher hidden-state tuple lengths differ: {len(student_hidden)} vs {len(teacher_hidden)}"
            )

        mask = token_mask.to(dtype=torch.bool)
        valid = mask.sum().clamp_min(1)
        losses = []

        for layer in self.rep_layers:
            idx = self._resolve_layer_index(layer, student_hidden)
            hs = student_hidden[idx]
            ht = teacher_hidden[idx].detach()
            if hs.shape != ht.shape:
                raise ValueError(f"Hidden-state shape mismatch at layer {layer}: {tuple(hs.shape)} vs {tuple(ht.shape)}")

            hs = F.normalize(hs.float(), p=2, dim=-1, eps=1e-6)
            ht = F.normalize(ht.float(), p=2, dim=-1, eps=1e-6)

            if self.rep_loss_type == "cosine":
                per_token = 1.0 - (hs * ht).sum(dim=-1)
            else:
                per_token = (hs - ht).pow(2).mean(dim=-1)

            losses.append((per_token * mask.to(per_token.dtype)).sum() / valid)

        return torch.stack(losses).mean()

    def _compute_loss_and_backward(self, batch, do_backward=True, n_micro_batches=1):
        """Compute hybrid SFT + representation warm-up loss."""
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        raw_loss_mask = batch["loss_mask"].to(self.device_name)
        ce_loss_mask = raw_loss_mask[:, 1:].reshape(-1)
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            labels = input_ids[:, 1:].contiguous()
            student_output = self.fsdp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )

            logits = student_output.logits
            shift_logits = logits[..., :-1, :].contiguous().view(-1, self.model.config.vocab_size)
            shift_labels = labels.contiguous().view(-1).to(shift_logits.device)
            token_ce = loss_fct(shift_logits, shift_labels)
            token_ce = token_ce * ce_loss_mask.to(token_ce.device)

            valid_token_this_rank = torch.sum(ce_loss_mask)
            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(valid_token_this_rank)
                dp_size = torch.distributed.get_world_size()
            else:
                dp_size = 1
            sft_loss = torch.sum(token_ce) / (valid_token_this_rank + 1e-8) * dp_size

            with torch.no_grad():
                teacher_output = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )

            if self.rep_response_only:
                rep_mask = raw_loss_mask.bool() & attention_mask.bool()
            else:
                rep_mask = attention_mask.bool()

            rep_loss = self._representation_loss(
                student_output.hidden_states,
                teacher_output.hidden_states,
                rep_mask,
            )
            total_loss = self.rep_alpha * sft_loss + self.rep_beta * rep_loss

            total_loss = total_loss / n_micro_batches
            sft_metric = sft_loss.detach() / n_micro_batches
            rep_metric = rep_loss.detach() / n_micro_batches

            if do_backward:
                total_loss.backward()

        return total_loss, sft_metric, rep_metric

    def training_step(self, batch: TensorDict):
        start_time = time.time()
        self.fsdp_model.train()
        self.teacher_model.eval()
        self.optimizer.zero_grad()

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_total = 0.0
        step_sft = 0.0
        step_rep = 0.0

        for micro_batch in micro_batches:
            total, sft, rep = self._compute_loss_and_backward(
                batch=micro_batch,
                n_micro_batches=n_micro_batches,
            )
            step_total += total.detach().item()
            step_sft += sft.item()
            step_rep += rep.item()

        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")

        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        self.lr_scheduler.step()

        metrics = torch.tensor([step_total, step_sft, step_rep], device=self.device_name)
        if is_cuda_available:
            torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.AVG)
        elif is_npu_available:
            torch.distributed.all_reduce(metrics)
            metrics /= self.device_mesh.size(0)

        return {
            "train/loss": metrics[0].item(),
            "train/sft_loss": metrics[1].item(),
            "train/rep_loss": metrics[2].item(),
            "train/lr(1e-3)": self.lr_scheduler.get_last_lr()[0] * 1e3,
            "train/time(s)": time.time() - start_time,
        }

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        self.teacher_model.eval()
        with torch.no_grad():
            total, _, _ = self._compute_loss_and_backward(batch, do_backward=False)
            if is_cuda_available:
                torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(total)
                total /= self.device_mesh.size(0)
        return total


def run_tcrw_sft(config):
    device_name = get_device_name()
    _, _, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    train_dataset = create_sft_dataset(
        config.data.train_files, config.data, tokenizer, max_samples=config.data.get("train_max_samples", -1)
    )
    val_dataset = create_sft_dataset(
        config.data.val_files, config.data, tokenizer, max_samples=config.data.get("val_max_samples", -1)
    )

    trainer = TCRWFSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    if config.get("representation_warmup", None) is None:
        raise ValueError(
            "Missing +representation_warmup.* Hydra arguments. Use the provided run_tcrw_warmup.sh launcher."
        )
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        print(OmegaConf.to_yaml(config.representation_warmup))
    run_tcrw_sft(config)


if __name__ == "__main__":
    main()
