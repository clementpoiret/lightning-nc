# Lightning Neural Compressor

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.md)

This repository contains the implementation of the Lightning Neural Compressor. The main goal of this project is to provide Pytorch Lightning callbacks to use Intel® Neural Compressor. The callbacks aim at compressing a neural network so that it can be used on edge devices (i.e., mobile phones, raspberry pi, etc.). This project is a work in progress and is not ready for production use.

This work is supported by Caminov (FRINGUANT).

## Current Status

The project is currently under development, starting with Quantization Aware Training, as the default callback has been deleted from Pytorch Lightning.

The project also supports Weight Pruning and should work at least with pruners related to the [`PytorchBasicPruner`](https://github.com/intel/neural-compressor/blob/d81269d2b261d39967605e17a89b5688ebaedbd1/neural_compressor/compression/pruner/pruners/basic.py#L29).

## Installation

To install the dependencies for this project, use the following command to use pypi:

```
pip install -U lightning-nc
```

or directly by cloning the main branch:

```bash
git clone https://github.com/clementpoiret/lightning-nc
cd lightning-nc
pip install -e .
```

## Usage

To use the Lightning Neural Compressor, import the callbacks from the `lightning_nc` module.

**WARNING:** Currently, the callbacks need the PyTorch model to be a `nn.Module` contained inside your `LightningModule`.
This is not a huge limitation as the refactoring is easy and straightforward, such as:

```python
import os

import lightning as L
import timm
import torch
import torch.nn.functional as F
from neural_compressor import QuantizationAwareTrainingConfig
from neural_compressor.config import Torch2ONNXConfig
from neural_compressor.training import WeightPruningConfig
from lightning_nc import QATCallback, WeightPruningCallback
from torch import Tensor, nn, optim, utils
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


# Define your main model here
class VeryComplexModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("best_pretrained_model",
                                          pretrained=True)

        self.mlp = nn.Sequential(nn.Linear(self.backbone.num_features, 128),
                                 nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.mlp(self.backbone(x))


# Then, define your LightningModule as usual
class Classifier(L.LightningModule):

    def __init__(self):
        super().__init__()

        # This is mandatory for the callbacks
        self.model = VeryComplexModel()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        # This is just to use MNIST images on a pretrained timm model, you can skip that
        x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(224, 224))

        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        return [optimizer]


clf = Classifier()

# setup data
dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset)
```

Now that everything is setup, the callbacks can be integrated into a PyTorch Lightning training routine:

```python
# Define the configs for Pruning and Quantization
q_config = QuantizationAwareTrainingConfig()
p_config = WeightPruningConfig([{
    "op_names": ["backbone.*"],
    "start_step": 1,
    "end_step": 100,
    "target_sparsity": 0.5,
    "pruning_frequency": 1,
    "pattern": "4x1",
    "min_sparsity_ratio_per_op": 0.,
    "pruning_scope": "global",
}])

callbacks = [
    QATCallback(config=q_config),
    WeightPruningCallback(config=p_config),
]

trainer = L.Trainer(accelerator="gpu",
                    strategy="auto",
                    limit_train_batches=100,
                    max_epochs=1,
                    callbacks=callbacks)
trainer.fit(model=clf, train_dataloaders=train_loader)
```

Models can now be saved eaily such as:

```python
clf.model.export(
    "model.onnx",
    Torch2ONNXConfig(
        dtype="int8",
        opset_version=17,
        quant_format="QOperator",  # or QDQ
        example_inputs=torch.randn(1, 3, 224, 224),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {
                0: "batch_size"
            },
            "output": {
                0: "batch_size"
            },
        },
    ))

```

## Contributing

If you would like to contribute to this project, please submit a pull request. All contributions are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.

