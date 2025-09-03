"""
TorchRec wrapper for distributed recommendation system training.

This module provides a wrapper class for TorchRec models that handles
distributed training, validation, checkpointing, and metrics logging.
"""
import itertools
import logging
import os
import random
import tempfile
import time
from typing import Dict, Optional, Tuple

import numpy as np
import ray
import torch
import torchmetrics as metrics
from ray.data.collate_fn import CollateFn, NumpyBatchCollateFn

from configs import RecsysConfig
from criteo import DatasetKey, convert_to_torchrec_batch_format

logger = logging.getLogger(__name__)

class TorchRecWrapper:
    """
    Wrapper class for TorchRec distributed recommendation model training.
    
    Handles model initialization, training loops, validation, checkpointing,
    and distributed training coordination using Ray Train.
    """
    
    def __init__(self, config: RecsysConfig):
        """
        Initialize TorchRec wrapper with configuration.
        
        Args:
            config: RecsysConfig object containing training parameters
        """
        from torchrec.distributed import TrainPipelineSparseDist

        self.config = config
        self._setup_reproducibility()
        
        # Initialize model and optimizer
        self.model = self.get_model()
        self.optimizer = self.get_optimizer()
        
        # Training state tracking
        self._train_epoch_idx = 0
        self._current_step = 0
        self._last_checkpoint_time = time.time()

        # Create training and validation pipelines
        device = ray.train.torch.get_device()
        self._train_pipeline = TrainPipelineSparseDist(self.model, self.optimizer, device)
        self._val_pipeline = TrainPipelineSparseDist(self.model, self.optimizer, device)

    def _setup_reproducibility(self) -> None:
        """Setup random seeds for reproducible training."""
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


    def get_model(self) -> torch.nn.Module:
        """
        Create and configure the distributed TorchRec model.
        
        Returns:
            Distributed model parallel TorchRec model
        """
        from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
        from torchrec.modules.embedding_configs import EmbeddingBagConfig
        from torchrec import EmbeddingBagCollection
        from torchrec.distributed.model_parallel import (
            DistributedModelParallel,
            get_default_sharders,
        )
        import torch.distributed as torch_dist
        from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
        from torchrec.distributed.planner.storage_reservations import (
            HeuristicalStorageReservation,
        )
        from torchrec.models.dlrm import DLRM, DLRM_DCN, DLRM_Projection, DLRMTrain
        from torchrec.optim.apply_optimizer_in_backward import (
            apply_optimizer_in_backward,
        )

        device = ray.train.torch.get_device()
        local_world_size = ray.train.get_context().get_local_world_size()
        global_world_size = ray.train.get_context().get_world_size()

        eb_configs = [
            EmbeddingBagConfig(
                name=f"t_{feature_name}",
                embedding_dim=self.config.embedding_dim,
                num_embeddings=self.config.num_embeddings_per_feature[feature_idx],
                feature_names=[feature_name],
            )
            for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
        ]
        sharded_module_kwargs = {}
        if self.config.over_arch_layer_sizes is not None:
            sharded_module_kwargs["over_arch_layer_sizes"] = self.config.over_arch_layer_sizes

        if self.config.interaction_type == "original":
            dlrm_model = DLRM(
                embedding_bag_collection=EmbeddingBagCollection(
                    tables=eb_configs, device=torch.device("meta")
                ),
                dense_in_features=len(DEFAULT_INT_NAMES),
                dense_arch_layer_sizes=self.config.dense_arch_layer_sizes,
                over_arch_layer_sizes=self.config.over_arch_layer_sizes,
                dense_device=device,
            )
        elif self.config.interaction_type == "dcn":
            dlrm_model = DLRM_DCN(
                embedding_bag_collection=EmbeddingBagCollection(
                    tables=eb_configs, device=torch.device("meta")
                ),
                dense_in_features=len(DEFAULT_INT_NAMES),
                dense_arch_layer_sizes=self.config.dense_arch_layer_sizes,
                over_arch_layer_sizes=self.config.over_arch_layer_sizes,
                dcn_num_layers=self.config.dcn_num_layers,
                dcn_low_rank_dim=self.config.dcn_low_rank_dim,
                dense_device=device,
            )
        elif self.config.interaction_type == "projection":
            raise NotImplementedError

            dlrm_model = DLRM_Projection(
                embedding_bag_collection=EmbeddingBagCollection(
                    tables=eb_configs, device=torch.device("meta")
                ),
                dense_in_features=len(DEFAULT_INT_NAMES),
                dense_arch_layer_sizes=args.dense_arch_layer_sizes,
                over_arch_layer_sizes=args.over_arch_layer_sizes,
                interaction_branch1_layer_sizes=args.interaction_branch1_layer_sizes,
                interaction_branch2_layer_sizes=args.interaction_branch2_layer_sizes,
                dense_device=device,
            )
        else:
            raise ValueError(
                "Unknown interaction option set. Should be original, dcn, or projection."
            )

        train_model = DLRMTrain(dlrm_model)
        embedding_optimizer = torch.optim.Adagrad
        # This will apply the Adagrad optimizer in the backward pass for the embeddings (sparse_arch). This means that
        # the optimizer update will be applied in the backward pass, in this case through a fused op.
        # TorchRec will use the FBGEMM implementation of EXACT_ADAGRAD. For GPU devices, a fused CUDA kernel is invoked. For CPU, FBGEMM_GPU invokes CPU kernels
        # https://github.com/pytorch/FBGEMM/blob/2cb8b0dff3e67f9a009c4299defbd6b99cc12b8f/fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops.py#L676-L678

        # Note that lr_decay, weight_decay and initial_accumulator_value for Adagrad optimizer in FBGEMM v0.3.2
        # cannot be specified below. This equivalently means that all these parameters are hardcoded to zero.
        optimizer_kwargs = {"lr": self.config.learning_rate, "eps": self.config.optimizer_eps}
        apply_optimizer_in_backward(
            embedding_optimizer,
            train_model.model.sparse_arch.parameters(),
            optimizer_kwargs,
        )
        planner = EmbeddingShardingPlanner(
            topology=Topology(
                local_world_size=local_world_size,
                world_size=global_world_size,
                compute_device=device.type,
            ),
            batch_size=self.config.train_batch_size,
            # If experience OOM, increase the percentage. see
            # https://pytorch.org/torchrec/torchrec.distributed.planner.html#torchrec.distributed.planner.storage_reservations.HeuristicalStorageReservation
            storage_reservation=HeuristicalStorageReservation(percentage=0.05),
        )
        plan = planner.collective_plan(
            train_model, get_default_sharders(), torch_dist.GroupMember.WORLD
        )

        model = DistributedModelParallel(
            module=train_model,
            device=device,
            plan=plan,
        )

        if self._is_main_process():
            for collectionkey, plans in model._plan.plan.items():
                logger.info(collectionkey)
                for table_name, plan in plans.items():
                    logger.info(table_name)
                    logger.info(plan)

        return model

    def get_optimizer(self) -> torch.optim.Optimizer:
        """
        Create combined optimizer for sparse and dense parameters.
        
        Returns:
            Combined optimizer with fused sparse optimizer and dense Adagrad optimizer
        """
        from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
        from torchrec.optim.optimizers import in_backward_optimizer_filter

        dense_optimizer = KeyedOptimizerWrapper(
            dict(in_backward_optimizer_filter(self.model.named_parameters())),
            lambda params: torch.optim.Adagrad(
                params, 
                lr=self.config.learning_rate, 
                eps=self.config.optimizer_eps
            ),
        )
        return CombinedOptimizer([self.model.fused_optimizer, dense_optimizer])

    def _get_collate_fn(self) -> Optional[CollateFn]:
        """
        Create collate function for TorchRec batch format conversion.
        
        Returns:
            Collate function that converts numpy batches to TorchRec format
        """
        from torchrec.datasets.utils import Batch

        class TorchRecCollateFn(NumpyBatchCollateFn):
            def __call__(self, batch: Dict[str, np.ndarray]) -> Batch:
                return convert_to_torchrec_batch_format(batch)

        return TorchRecCollateFn()

    def get_train_dataloader(self):
        """
        Create training dataloader from Ray dataset shard.
        
        Returns:
            Iterator over training batches in TorchRec format
        """
        ds_iterator = ray.train.get_dataset_shard(DatasetKey.TRAIN)

        # Apply training row limit if configured
        if (self.config.limit_training_rows is not None and 
            self.config.limit_training_rows > 0):
            ds_iterator = ds_iterator.limit(self.config.limit_training_rows)
            if self._is_main_process():
                logger.info(f"Training limited to {self.config.limit_training_rows} rows")

        return iter(
            ds_iterator.iter_torch_batches(
                batch_size=self.config.train_batch_size,
                collate_fn=self._get_collate_fn(),
                drop_last=True,
            )
        )

    def get_val_dataloader(self):
        """
        Create validation dataloader from Ray dataset shard.
        
        Returns:
            Iterator over validation batches in TorchRec format
        """
        ds_iterator = ray.train.get_dataset_shard(DatasetKey.VALID)
        
        return iter(
            ds_iterator.iter_torch_batches(
                batch_size=self.config.validation_batch_size,
                collate_fn=self._get_collate_fn(),
                drop_last=False,
            )
        )

    def _validate_step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            # https://github.com/pytorch/torchrec/blob/1ab1381c663cd25e03731122cfd70615ae964bf4/torchrec/models/dlrm.py#L902
            # the model is returning
            # loss, (loss.detach(), logits.detach(), batch.labels.detach())
            loss, (_, logits, labels) = self.model(batch)
        return loss, logits, labels

    def _train_epoch(self) -> None:
        if self._is_main_process():
            logger.info(f"Training starting @ epoch={self._train_epoch_idx}")

        train_dataloader = self.get_train_dataloader()

        # Skip through batches if we restored to a middle of the epoch.
        # TODO: Compare this baseline to the data checkpointing approach once we have it.

        total_train_loss = 0.0
        num_batches = 0
        start_time = time.time()
        last_log_time = start_time
        last_log_batches = 0
        # Track validation time to exclude it from throughput calculation
        accumulated_validation_time = 0.0
        world_size = ray.train.get_context().get_world_size()
        total_time = 0

        for it in itertools.count(1):
            try:
                if not self.config.skip_train_step:
                    batch_loss, _logits, _labels = self._train_pipeline.progress(train_dataloader)
                    try:
                        loss_value = batch_loss.item()
                    except Exception:
                        loss_value = float(batch_loss)
                    total_train_loss += loss_value
                    num_batches += 1
                    self._current_step += 1
                    
                    # Log individual batch loss and running average during training
                    if (self._is_main_process() and 
                        self.config.log_metrics_every_n_steps > 0 and 
                        num_batches % self.config.log_metrics_every_n_steps == 0):

                        current_time = time.time()
                        current_avg_loss = total_train_loss / num_batches
                        
                        # Calculate real-time throughput since last log
                        time_since_last_log = current_time - last_log_time - accumulated_validation_time
                        total_time += time_since_last_log
                        # reset
                        accumulated_validation_time = 0.0
                        batches_since_last_log = num_batches - last_log_batches
                        if time_since_last_log > 0:
                            instant_throughput_batches = batches_since_last_log / time_since_last_log
                            instant_throughput_samples = (batches_since_last_log * self.config.train_batch_size) / time_since_last_log
                        else:
                            instant_throughput_batches = 0
                            instant_throughput_samples = 0
                        
                        # Calculate cumulative throughput from epoch start
                        
                        if total_time > 0:
                            cumulative_throughput_batches = num_batches / total_time
                            cumulative_throughput_samples = (num_batches * self.config.train_batch_size) / total_time
                        else:
                            cumulative_throughput_batches = 0
                            cumulative_throughput_samples = 0
                        
                        # Globalized metrics
                        global_instant_samples = instant_throughput_samples * world_size
                        global_cumulative_batches = cumulative_throughput_batches * world_size
                        global_cumulative_samples = cumulative_throughput_samples * world_size
                        elapsed_time = current_time - start_time

                        logger.info({
                            # Namespaced training metrics
                            "train/instant_throughput_batches_per_sec": instant_throughput_batches * world_size,
                            "train/instant_throughput_samples_per_sec": global_instant_samples,
                            "train/cumulative_throughput_batches_per_sec": global_cumulative_batches,
                            "train/cumulative_throughput_samples_per_sec": global_cumulative_samples,
                            # Metrics aligned with dlrm_main
                            "loss": batch_loss.item(),
                            "running_avg_loss": current_avg_loss,
                            "throughput_samples_per_sec": global_instant_samples,
                            "iteration": num_batches,
                            "total_samples_processed": num_batches * self.config.train_batch_size * world_size,
                            "elapsed_time": elapsed_time,
                            "epoch": self._train_epoch_idx,
                        })
                        logger.info(f"Epoch {self._train_epoch_idx}, Step {num_batches}: "
                                f"Batch loss: {batch_loss.item():.4f}, "
                                f"Running avg: {current_avg_loss:.4f}, "
                                f"Throughput: {(instant_throughput_samples * ray.train.get_context().get_world_size()):.1f} samples/sec")
                        
                        # Update tracking variables
                        last_log_time = current_time
                        last_log_batches = num_batches

                    # Validate every N steps
                    if (self.config.validate_every_n_steps > 0 and 
                        num_batches % self.config.validate_every_n_steps == 0):
                        
                        if self._is_main_process():
                            logger.info(f"Running validation at step {self._current_step}")
                        
                        validation_start_time = time.time()
                        self._validate()
                        validation_end_time = time.time()
                        accumulated_validation_time += validation_end_time - validation_start_time
                        self._train_pipeline._model.train()

                    # Save checkpoint every N minutes
                    if self._should_save_checkpoint_at_time(self._current_step):
                        self._save_checkpoint(self._train_epoch_idx, self._current_step)
            except StopIteration:
                break

        # Log final epoch average training loss and throughput
        if num_batches > 0 and self._is_main_process():
            end_time = time.time()
            total_training_time = end_time - start_time
            
            avg_train_loss = total_train_loss / num_batches
            final_throughput_batches = num_batches / total_training_time if total_training_time > 0 else 0
            final_throughput_samples = (num_batches * self.config.train_batch_size) / total_training_time if total_training_time > 0 else 0
            world_size = ray.train.get_context().get_world_size()
            logger.info({
                "train/epoch_avg_loss": avg_train_loss,
                "train/epoch_throughput_batches_per_sec": final_throughput_batches * world_size,
                "train/epoch_throughput_samples_per_sec": final_throughput_samples * world_size,
                "train/epoch_duration_sec": total_training_time,
                "train/total_samples_processed": num_batches * self.config.train_batch_size * world_size,
                "epoch": self._train_epoch_idx,
                # Metrics aligned with dlrm_main
                "epoch_final_throughput_samples_per_sec": final_throughput_samples * world_size,
                "epoch_total_iterations": num_batches,
                "epoch_total_samples": num_batches * self.config.train_batch_size * world_size,
                "epoch_total_time": total_training_time,
            })
            logger.info(f"Epoch {self._train_epoch_idx} - Average training loss: {avg_train_loss:.4f}")
            logger.info(f"Epoch {self._train_epoch_idx} - Final throughput: {final_throughput_samples:.1f} samples/sec "
                    f"({final_throughput_batches:.1f} batches/sec) over {total_training_time:.1f}s")

        self._train_epoch_idx += 1

    def _is_main_process(self) -> bool:
        return ray.train.get_context().get_world_rank() == 0

    def _save_checkpoint(self, epoch: int, step: int, is_final: bool = False) -> None:
        """Save model checkpoint using Ray Train's checkpoint functionality."""
        logger.info(f"Saving checkpoint at epoch {epoch}, step {step}")
        
        # Create checkpoint data
        checkpoint_data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": self.config.__dict__,
        }

        metrics_to_report = {"epoch": epoch, "step": step}
        if is_final:
            metrics_to_report["final_checkpoint"] = True

        with tempfile.TemporaryDirectory() as temp_dir:
            rank = ray.train.get_context().get_world_rank()
            torch.save(checkpoint_data, os.path.join(temp_dir, f"checkpoint_{rank}.pt"))
            checkpoint = ray.train.Checkpoint.from_directory(temp_dir)
            ray.train.report(checkpoint=checkpoint, metrics=metrics_to_report)
        
        # Update last checkpoint time for time-based checkpointing
        self._last_checkpoint_time = time.time()
        
        logger.info(f"Checkpoint reported to Ray Train at epoch {epoch}, step {step}")

    def _should_save_checkpoint_at_time(self, step: int) -> bool:
        """Check if we should save a checkpoint based on time elapsed."""
        if self.config.save_checkpoint_every_n_mins <= 0:
            return False
        
        current_time = time.time()
        time_elapsed_mins = (current_time - self._last_checkpoint_time) / 60.0
        return time_elapsed_mins >= self.config.save_checkpoint_every_n_mins

    def _should_save_checkpoint_at_epoch(self, epoch: int) -> bool:
        """Check if we should save a checkpoint at this epoch."""
        return (self.config.save_checkpoint_every_n_epochs > 0 and 
                (epoch + 1) % self.config.save_checkpoint_every_n_epochs == 0)

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Load model checkpoint from a file."""
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_dir}")
            
        if self._is_main_process():
            logger.info(f"Loading checkpoint from {checkpoint_dir}")
        
        # Load checkpoint data
        checkpoint_data = torch.load(
            os.path.join(checkpoint_dir, f"checkpoint_{ray.train.get_context().get_world_rank()}.pt"),
            map_location="cpu",
            weights_only=False,
        )
        
        # Restore model state
        self.model.load_state_dict(checkpoint_data["model_state_dict"])
        
        # Restore optimizer state
        self.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        
        # Restore training state
        self._train_epoch_idx = checkpoint_data["epoch"]
        self._current_step = checkpoint_data["step"]
        
        if self._is_main_process():
            logger.info(f"Checkpoint loaded successfully. Resuming from epoch {self._train_epoch_idx}, step {self._current_step}")

    @classmethod
    def from_checkpoint(cls, config: RecsysConfig, checkpoint_dir: str) -> 'TorchRecWrapper':
        """Create a TorchRecWrapper instance and load from checkpoint."""
        wrapper = cls(config)
        wrapper.load_checkpoint(checkpoint_dir)
        return wrapper

    def _validate(self) -> Dict[str, float]:
        if self._is_main_process():
            logger.info(
                f"Validation starting @ epoch={self._train_epoch_idx}, step={self._current_step}"
            )

        val_dataloader = self.get_val_dataloader()
        self._val_pipeline._model.eval()

        total_loss_sum = 0.0
        total_sample_count = 0
        
        # Initialize AUROC metric for validation
        device = ray.train.torch.get_device()
        auroc_metric = metrics.AUROC(task='binary').to(device)

        with torch.no_grad():
            while True:
                try:
                    batch_loss, logits, labels = self._val_pipeline.progress(val_dataloader)
                    preds = torch.sigmoid(logits)
                    auroc_metric(preds, labels)
                    try:
                        loss_value = batch_loss.item()
                    except Exception:
                        loss_value = float(batch_loss)
                    batch_size = labels.shape[0] if hasattr(labels, "shape") else len(labels)
                    total_loss_sum += loss_value * float(batch_size)
                    total_sample_count += int(batch_size)

                except StopIteration:
                    break

        # Compute validation metrics
        validation_loss = (total_loss_sum / float(max(1, total_sample_count))) if total_sample_count > 0 else 0.0
        auroc_result = auroc_metric.compute().item()
        
        results = {"val/loss": validation_loss}
        
        results["val/auroc"] = auroc_result
        
        if self._is_main_process():
            logger.info({
                "val/loss": validation_loss,
                "val/auroc": auroc_result,
                "epoch": self._train_epoch_idx
            })
            logger.info(f"Epoch {self._train_epoch_idx} - Validation loss: {validation_loss:.4f}, AUROC: {auroc_result:.4f}")

        return results

    def run(self) -> None:
        for epoch in range(self.config.num_epochs):
            self._train_epoch()
            if not self.config.skip_validation_at_epoch_end:
                self._validate()
            
            # Save checkpoint every N epochs
            if self._should_save_checkpoint_at_epoch(epoch):
                self._save_checkpoint(epoch, self._current_step)
        
        # Save final checkpoint if enabled
        if self.config.save_checkpoint_at_end:
            self._save_checkpoint(self._train_epoch_idx - 1, self._current_step, is_final=True)


def train_loop(config_dict: dict) -> None:
    """
    Ray Train loop function for distributed training.
    
    This function is called by Ray Train workers to execute the training loop.
    It handles checkpoint restoration and delegates to the TorchRecWrapper.
    
    Args:
        config_dict: Dictionary containing RecsysConfig parameters
    """
    # Convert dictionary back to RecsysConfig
    recsys_config = RecsysConfig()
    for key, value in config_dict.items():
        setattr(recsys_config, key, value)

    # Handle checkpoint restoration
    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        # Resuming from worker failure
        with checkpoint.as_directory() as temp_checkpoint_dir:
            wrapper = TorchRecWrapper.from_checkpoint(recsys_config, temp_checkpoint_dir)
    elif recsys_config.start_from_checkpoint_path:
        # Starting from a specified checkpoint
        checkpoint = ray.train.Checkpoint(recsys_config.start_from_checkpoint_path)
        logger.info(f"Starting from checkpoint: {recsys_config.start_from_checkpoint_path}")
        with checkpoint.as_directory() as temp_checkpoint_dir:
            logger.info(f"Loading checkpoint from: {temp_checkpoint_dir}")
            wrapper = TorchRecWrapper.from_checkpoint(recsys_config, temp_checkpoint_dir)
    else:
        # Fresh training start
        wrapper = TorchRecWrapper(recsys_config)

    # Run training - the wrapper handles all epoch logic internally
    wrapper.run()