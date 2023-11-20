"""
Implements a callback to finetune a model.
This callback enables advanced training setups,
allowing to combine finetuning and QAT / Pruning.
"""

import re
from typing import Dict

from lightning.pytorch import Callback, LightningModule, Trainer


class FinetuningCallback(Callback):
    """Finetuning callback for PyTorch Lightning

    Example config:

    ```python
    config = {
        ".*last_layer.*": {
            "start_epoch": 0,
            "end_epoch": -1,
        },
        ".*backbone.*": {
            "start_epoch": 16,
            "end_epoch": -1,
        },
    }
    ```

    Attributes
    ----------
    config: Dict
        Configuration for finetuning.

    Methods
    -------
    on_train_epoch_start(trainer, pl_module)
        Called when the train epoch begins.
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config

    def setup(self, trainer: Trainer, pl_module: LightningModule,
              stage: str) -> None:
        self.max_epochs = trainer.max_epochs + 1

    def on_train_epoch_start(self, trainer: Trainer,
                             pl_module: LightningModule) -> None:
        for name, module in pl_module.named_modules():
            for pattern, info in self.config.items():
                if re.match(pattern, name):
                    _max_epochs = self.max_epochs if info[
                        "end_epoch"] == -1 else info["end_epoch"]
                    if info["start_epoch"] <= trainer.current_epoch < _max_epochs:
                        print(f"Finetuning {name}")
                        module.train()
                        module.apply(lambda x: x.requires_grad_())
                    else:
                        print(f"Freezing {name}")
                        module.eval()
                        module.apply(lambda x: x.requires_grad_(False))
