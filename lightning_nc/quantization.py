"""Callbacks for PyTorch Lightning
It integrates the compression algorithms from Intel(R) Neural Compressor
into PyTorch Lightning framework. The compression algorithms are implemented
as callbacks. Most of the code is located at https://github.com/intel/neural-compressor/.
"""

from typing import Optional

from lightning.pytorch import Callback, LightningModule, Trainer
from neural_compressor import QuantizationAwareTrainingConfig
from neural_compressor.adaptor.pytorch import PyTorchAdaptor
from neural_compressor.config import Options, options
from neural_compressor.model.model import BaseModel, Model
from torch.utils.data import DataLoader


class QATCallback(Callback):
    """Quantization aware training callback for PyTorch Lightning

    Attributes
    ----------
    config: QuantizationAwareTrainingConfig
        Configuration for quantization aware training.
    adaptor: PyTorchAdaptor
        Adaptor of PyTorch framework, all PyTorch API is in this class.

    Methods
    -------
    setup(trainer, pl_module, stage)
        Setup the callback.
    teardown(trainer, pl_module, stage)
        Teardown the callback.
    """

    def __init__(self,
                 config: QuantizationAwareTrainingConfig,
                 backend: str = "default",
                 quant_format: str = "default",
                 options: Options = options,
                 dataloader: Optional[DataLoader] = None) -> None:
        """
        Parameters
        ----------
        config : QuantizationAwareTrainingConfig
            Configuration for quantization aware training.
        backend : str, optional
            Backend of quantization aware training, by default "default"
        quant_format : str, optional
            Format of quantization in ONNXRuntime, by default "default"
        options : Options, optional
            Options for quantization aware training
        dataloader : Optional[DataLoader], optional
            Dataloader for accuracy calibration, by default None
        """
        super().__init__()
        self.config = config
        self.dataloader = dataloader

        # Taken from
        # https://github.com/intel/neural-compressor/blob/d81269d2b261d39967605e17a89b5688ebaedbd1/neural_compressor/training.py#L130
        framework_specific_info = {
            "device": config.device,
            "random_seed": options.random_seed,
            "workspace_path": options.workspace,
            "q_dataloader": None,
            "backend": backend,
            "format": quant_format,
            "approach": config.approach,
        }

        # WARNING: May change in a future version of Intel(R) Neural Compressor
        framework_specific_info["qat_optype_wise"] = config.op_type_dict
        framework_specific_info["qat_op_wise"] = config.op_name_dict

        self.adaptor = PyTorchAdaptor(framework_specific_info)

    def setup(self, trainer: Trainer, pl_module: LightningModule,
              stage: str) -> None:
        """Setup the callback by converting the model.
        Original hook defined here: https://github.com/intel/neural-compressor/blob/d81269d2b261d39967605e17a89b5688ebaedbd1/neural_compressor/compression/callbacks.py#L194
        """
        if stage == "fit":
            if isinstance(pl_module.model, BaseModel):
                self.adaptor.model = pl_module.model.model = Model(
                    pl_module.model.model, conf=self.config)
            else:
                self.adaptor.model = pl_module.model = Model(pl_module.model,
                                                             conf=self.config)

            self.adaptor._pre_hook_for_qat(dataloader=self.dataloader)

    def teardown(self, trainer: Trainer, pl_module: LightningModule,
                 stage: str) -> None:
        """Teardown the callback.
        Original hook defined here: https://github.com/intel/neural-compressor/blob/d81269d2b261d39967605e17a89b5688ebaedbd1/neural_compressor/compression/callbacks.py#L195
        """
        if stage == "fit":
            self.adaptor._post_hook_for_qat()
