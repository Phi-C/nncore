import functools
from typing import List, Dict, Any
import torch
import torch.nn as nn
from torchinfo import summary


class TorchInfoSummarizer:
    """
    A class to summarize PyTorch models using torchinfo by tracking forward passes.
    
    Example:
        >> from torchvision import models

        >> model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        >> summarizer = TorchInfoSummarizer()
        >> model.forward = summarizer.mock_model_forward(model)
        >> model.layer1.forward = summarizer.mock_model_forward(model.layer1)
        >> model(torch.rand(1, 3, 224, 224))
        >> summarizer.get_summary()
    """

    def __init__(self):
        self._tracked_models: List[torch.nn.Module] = []
        self._original_forwards: List[Any] = []
        self._forward_inputs: List[Dict[str, Any]] = []

    def mock_model_forward(self, model: torch.nn.Module) -> callable:
        """Wrap a model's forward method to track its inputs.

        Args:
            model: The PyTorch module to track

        Returns:
            A wrapped forward method that records inputs
        """
        if model in self._tracked_models:
            return model.forward

        self._tracked_models.append(model)
        original_forward = model.forward
        self._original_forwards.append(original_forward)

        @functools.wraps(original_forward)
        def wrapped_forward(*args, **kwargs):
            # Store the input data
            self._forward_inputs.append(
                {
                    "args": args,
                    "kwargs": kwargs,
                }
            )
            # Call the original forward method directly
            return original_forward(*args, **kwargs)

        return wrapped_forward

    def get_summary(self) -> None:
        """Generate and print model summaries for all tracked forward passes."""
        for model, original_forward, inputs in zip(
            self._tracked_models, self._original_forwards, self._forward_inputs
        ):
            # Temporarily restore original forward for accurate summary
            model.forward = original_forward
            try:
                summary(model, input_data=inputs["args"], **inputs["kwargs"])
            finally:
                # Re-wrap the forward method
                model.forward = self.mock_model_forward(model)


def shape_infer_hook(name):
    """A closure that captures the module name"""

    def hook(module, input, output):
        if len(list(module.children())) == 0:
            print(f"Shape of output to {name} is {output.shape}.")

    return hook


def register_shape_infer_hook(net: nn.Module, name_list: List[str] = None):
    """
    Register forward hooks for all layers in a network.

    Example:
        >> import torch
        >> from torchvision.models import resnet18

        >> model = resnet18(pretrained=True)
        >> model.eval()
        >> inp = torch.randn(3, 3, 224, 224)

        >> register_shape_infer_hook(model)
        >> model(inp)
    """
    for name, layer in net.named_modules():
        if name_list is None or name in name_list:
            layer.register_forward_hook(shape_infer_hook(name))

