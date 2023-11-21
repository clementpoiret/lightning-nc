"""Callbacks for PyTorch Lightning
It integrates the compression algorithms from Intel(R) Neural Compressor
into PyTorch Lightning framework. The compression algorithms are implemented
as callbacks. Most of the code is located at https://github.com/intel/neural-compressor/.
"""

import logging
from typing import Any, Optional

from lightning.pytorch import Callback, LightningModule, Trainer
from neural_compressor.compression.pruner.model_slim.pattern_analyzer import \
    SelfMHASearcher
from neural_compressor.compression.pruner.pruners import get_pruner
from neural_compressor.compression.pruner.utils import (get_sparsity_ratio,
                                                        parse_to_prune,
                                                        process_config)
from neural_compressor.model.model import BaseModel, Model
from neural_compressor.training import WeightPruningConfig
from torch.optim import Optimizer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class WeightPruningCallback(Callback):
    """Weight pruning callback for PyTorch Lightning
    It basically calls Intel(R) Neural Compressor's hooks
    defined in their `BasePruning` class.
    More details can be found at:
    https://github.com/intel/neural-compressor/blob/d81269d2b261d39967605e17a89b5688ebaedbd1/neural_compressor/compression/pruner/pruning.py#L55

    Attributes
    ----------
    config: WeightPruningConfig
        Configuration for weight pruning.
    dataloader: Optional[DataLoader]
        Dataloader for accuracy calibration.

    Methods
    -------
    setup(trainer, pl_module, stage)
        Setup the callback.
    on_train_start(trainer, pl_module)
        Called when the train begins. Generate the pruners.
    """

    def __init__(self,
                 config: WeightPruningConfig = None,
                 dataloader: Optional[DataLoader] = None) -> None:
        """Initialize the callback.

        Parameters
        ----------
        config : WeightPruningConfig, optional
            Configuration for weight pruning, by default None
        dataloader : Optional[DataLoader], optional
            Dataloader for accuracy calibration, by default None
        """
        super().__init__()
        self.config = config
        self.dataloader = dataloader

        self.pruners_info = process_config(config)
        self.pruners = []

    def _generate_pruners(self, model: Model) -> None:
        """Generate the pruners.
        Original function copied taken from:
        https://github.com/intel/neural-compressor/blob/d81269d2b261d39967605e17a89b5688ebaedbd1/neural_compressor/compression/callbacks.py#L241

        Parameters
        ----------
        model : Model
            Model to be pruned.
        """
        for info in self.pruners_info:
            if "mha" in info["pattern"]:
                # head pruning
                pa_obj = SelfMHASearcher(model.model)
                modules, _ = pa_obj.search(split_qkv_ffn=False)
                modules = pa_obj.obtain_mha_module(modules)
                modules = pa_obj.from_layer_name_to_object(modules)
                if len(modules) == 0:
                    logger.warning(
                        "one pruner hooks no mha modules, please have a check")
                self.pruners.append(get_pruner(info, modules))
            else:
                # original pruning types, e.g NxM or N:M
                modules = parse_to_prune(info, model.model)
                if modules == {}:
                    logger.warning(
                        "one pruner hooks no layers, please have a check")

                self.pruners.append(get_pruner(info, modules))
                info["modules"] = [key for key in modules.keys()]
                info["len_of_modules"] = len(info["modules"])

                logger.info(info)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str):
        if stage == "fit":
            print("setup weight pruning callback")
            if isinstance(pl_module.model, BaseModel):
                pl_module.model.model = Model(pl_module.model.model,
                                              conf=self.config)
            else:
                pl_module.model = Model(pl_module.model, conf=self.config)

    def on_train_start(self, trainer: Trainer,
                       pl_module: LightningModule) -> None:

        self._generate_pruners(pl_module.model)

        for pruner in self.pruners:
            pruner.on_train_begin(dataloader=self.dataloader)

    def on_train_epoch_start(self, trainer: Trainer,
                             pl_module: LightningModule) -> None:
        current_epoch = trainer.current_epoch

        for pruner in self.pruners:
            pruner.on_epoch_begin(current_epoch)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        for pruner in self.pruners:
            pruner.on_step_begin(batch_idx)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for pruner in self.pruners:
            pruner.on_step_end()

    def on_train_epoch_end(self, trainer: Trainer,
                           pl_module: LightningModule) -> None:
        for pruner in self.pruners:
            pruner.on_epoch_end()

    def on_before_optimizer_step(self, trainer: Trainer,
                                 pl_module: LightningModule,
                                 optimizer: Optimizer) -> None:
        for pruner in self.pruners:
            # WARNING: This has to be tested. I'm not sure if this is the
            # correct way to do it. One this is for sure: if we call this
            # before step == start_step, in cases where we have params that
            # do not require grad, pruners based on gradients will fail to
            # compute their scores.
            if pruner.start_step <= trainer.global_step < pruner.end_step:
                pruner.on_before_optimizer_step()

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                           outputs: Any, batch: Any, batch_idx: int) -> None:
        # if batch_idx % trainer.accumulate_grad_batches == 0
        for pruner in self.pruners:
            # WARNING: Same thing as above. Because on_after_optimizer_step
            # keeps track of the global step, we can sync it with the trainer
            # global step to avoid an incorrect step count caused by our fix.
            if pruner.start_step <= trainer.global_step < pruner.end_step:
                pruner.on_after_optimizer_step()
            pruner.global_step = trainer.global_step

    def on_train_end(self, trainer: Trainer,
                     pl_module: LightningModule) -> None:
        for pruner in self.pruners:
            pruner.on_train_end()

        get_sparsity_ratio(self.pruners, pl_module.model)

    def on_validation_epoch_start(self, trainer: Trainer,
                                  pl_module: LightningModule) -> None:
        for pruner in self.pruners:
            pruner.on_before_eval()

    def on_validation_epoch_end(self, trainer: Trainer,
                                pl_module: LightningModule) -> None:
        for pruner in self.pruners:
            pruner.on_after_eval()
