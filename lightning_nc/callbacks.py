import logging
from typing import Any, Optional

from lightning.pytorch import Callback, LightningModule, Trainer
from neural_compressor import QuantizationAwareTrainingConfig
from neural_compressor.adaptor.pytorch import PyTorchAdaptor
from neural_compressor.compression.pruner.model_slim.pattern_analyzer import \
    SelfMHASearcher
from neural_compressor.compression.pruner.pruners import get_pruner
from neural_compressor.compression.pruner.utils import (get_sparsity_ratio,
                                                        parse_to_prune,
                                                        process_config)
from neural_compressor.config import Options, options
from neural_compressor.model.model import Model
from neural_compressor.training import WeightPruningConfig
from torch.utils.data import DataLoader
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class QATCallback(Callback):

    def __init__(self,
                 config: QuantizationAwareTrainingConfig,
                 backend: str = "default",
                 quant_format: str = "default",
                 options: Options = options,
                 dataloader: Optional[DataLoader] = None) -> None:
        super().__init__()
        self.config = config
        self.dataloader = dataloader

        framework_specific_info = {
            "device": config.device,
            "random_seed": options.random_seed,
            "workspace_path": options.workspace,
            "q_dataloader": None,
            "backend": backend,
            "format": quant_format,
            "approach": config.approach,
        }
        print("QATCallback: ", framework_specific_info)

        # WARNING: May change in a future version of Intel(R) Neural Compressor
        framework_specific_info["qat_optype_wise"] = config.op_type_dict
        framework_specific_info["qat_op_wise"] = config.op_name_dict

        self.adaptor = PyTorchAdaptor(framework_specific_info)

    def on_train_start(self, trainer: Trainer,
                       pl_module: LightningModule) -> None:
        self.adaptor.model = pl_module.model = Model(pl_module.model,
                                                     conf=self.config)
        self.adaptor._pre_hook_for_qat(dataloader=self.dataloader)

    def on_train_end(self, trainer: Trainer,
                     pl_module: LightningModule) -> None:
        self.adaptor._post_hook_for_qat()


# TODO: before and after eval


class WeightPruningCallback(Callback):

    def __init__(self,
                 config: WeightPruningConfig = None,
                 dataloader: Optional[DataLoader] = None) -> None:
        super().__init__()
        self.config = config
        self.dataloader = dataloader
        self.pruners_info = process_config(config)
        self.model = None
        self.pruners = []
        self.local_step = 0

    def _generate_pruners(self, model: Model) -> None:
        for info in self.pruners_info:
            if "mha" in info["pattern"]:
                # head pruning
                pa_obj = SelfMHASearcher(self.model.model)
                modules, _ = pa_obj.search(split_qkv_ffn=False)
                modules = pa_obj.obtain_mha_module(modules)
                modules = pa_obj.from_layer_name_to_object(modules)
                if len(modules) == 0:
                    logger.warning(
                        "one pruner hooks no mha modules, please have a check")
                self.pruners.append(get_pruner(info, modules))
            else:
                # original pruning types, e.g NxM or N:M
                modules = parse_to_prune(info, self.model.model)
                if modules == {}:
                    logger.warning(
                        "one pruner hooks no layers, please have a check")

                self.pruners.append(get_pruner(info, modules))
                info["modules"] = [key for key in modules.keys()]
                info["len_of_modules"] = len(info["modules"])
                logger.info(info)

    def on_train_start(self, trainer: Trainer,
                       pl_module: LightningModule) -> None:
        current_epoch = trainer.current_epoch

        if current_epoch == 0:
            self.model = pl_module.model = Model(pl_module.model,
                                                 conf=self.config)
            self._generate_pruners(self.model)

        for pruner in self.pruners:
            pruner.on_train_begin(dataloader=self.dataloader)

    def on_train_epoch_start(self, trainer: Trainer,
                             pl_module: LightningModule) -> None:
        current_epoch = trainer.current_epoch

        for pruner in self.pruners:
            pruner.on_epoch_begin(current_epoch)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        for pruner in self.pruners:
            pruner.on_step_begin(self.local_step)

        self.local_step += 1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for pruner in self.pruners:
            pruner.on_step_end()

        self.local_step = 0

    def on_train_epoch_end(self, trainer: Trainer,
                           pl_module: LightningModule) -> None:
        for pruner in self.pruners:
            pruner.on_epoch_end()

    def on_before_optimizer_step(self, trainer: Trainer,
                                 pl_module: LightningModule,
                                 optimizer: Optimizer) -> None:
        for pruner in self.pruners:
            pruner.on_before_optimizer_step()

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                           outputs: Any, batch: Any, batch_idx: int) -> None:
        # if batch_idx % trainer.accumulate_grad_batches == 0
        for pruner in self.pruners:
            pruner.on_after_optimizer_step()

    def on_train_end(self, trainer: Trainer,
                     pl_module: LightningModule) -> None:
        for pruner in self.pruners:
            pruner.on_train_end()

        get_sparsity_ratio(self.pruners, self.model)
