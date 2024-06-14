from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface


__all__ = [
    "ResNet",
    "ResNet18_Weights"
]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
       
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.add_relu_FF = torch.ao.nn.quantized.FloatFunctional()

    def modules_to_fuse(self, prefix):
        """
        This function takes a 'prefix' and creates a list of tuples containing
        names of modules to fuse. These are mostly convolutional layers followed
        by batch normalization and ReLU activation. Depending on downsampling
        feature flagged modules from downsample layer may also be included.

        Args:
            prefix (str): The `prefix` parameter specifies the prefix to be added
                to each name of the modules being fused. This is used to create
                meaningful names for the modules that are combined and will be
                useful during debugging.

        Returns:
            list: The output of the given function is a list containing multiple
            lists of strings representing the modules to be fused together.

        """
        modules_to_fuse_ = []
        modules_to_fuse_.append([f'{prefix}.conv1', f'{prefix}.bn1', f'{prefix}.relu1'])
        modules_to_fuse_.append([f'{prefix}.conv2', f'{prefix}.bn2'])
        if self.downsample:
            modules_to_fuse_.append([f'{prefix}.downsample.0', f'{prefix}.downsample.1'])

        return modules_to_fuse_

    def forward(self, x: Tensor) -> Tensor:
        """
        This function performs a convolutional layer using two convolutional layers
        with batch normalization and ReLU activation functions between them. It
        also includes an optional downsampling step.

        Args:
            x (Tensor): In this code snippet `x` is just an input tensor that is
                passed through a series of operations such as conv1 & batch
                normalization (bn1). No changes are made to x.

        Returns:
            Tensor: The output returned by the function "forward" is "out".

        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add_relu_FF.add_relu(out, identity)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
    
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )


        return nn.Sequential(*layers)

    def modules_to_fuse(self):
        """
        This function constructs a list of tuples containing the names of PyTorch
        modules to fuse (e.g., "conv1", "bn1", etc.), based on the structure of
        the neural network defined using the `self` context (i.e., an instance of
        a class with attributes `layer1`, `layer2`, `layer3`, and `layer4`). Specifically:
        - It first initializes an empty list `modules_to_fuse_`.
        - Then it appends a tuple containing the names of modules to fuse for the
        first layer (`conv1`, `bn1`, `relu`).
        Next is a loop over the remaining layers (`layer1`, `layer2`, and `layer3`)
        using an explicit list comprehension and the eval() function to extract
        each layer as an instance of some class (probably self.layerX):
        - for each layer (in ['layer1', 'layer2', 'layer3', 'layer4'])
           obtains its module's modules to fuse (self.layerX[block_nb].modules_to_fuse(
        prefix ) ) by running eval() on the str() representation of the layer.
        - extends the initial list with each iteration result.
        Finally it returns a list of tuples containing the names of PyTorch modules
        to fuse after the loop completes its iteration over self.layer1 up to and
        including self.layer4 .
        In sum: it defines which PyTorch modules to fuse for the layers (up until
        layer 4) within the given neural network object instance so that they can
        be efficiently executed on various devices supported by FusePool2

        Returns:
            list: The function "modules_to_fuse" takes a self-object as input and
            returns a list of lists with the names of the layers (as strings) that
            are to be fused together. The list of layers is generated by iterating
            through the layers of the model and fetching their module fusion
            information using eval() to fetch attributes of the layers using string
            names

        """
        modules_to_fuse_ = []
        modules_to_fuse_.append(['conv1', 'bn1', 'relu'])

        for layer_str in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = eval(f'self.{layer_str}')
            for block_nb in range(len(layer)):
                prefix = f'{layer_str}.{block_nb}'
                modules_to_fuse_layer = layer[block_nb].modules_to_fuse(prefix)
                modules_to_fuse_.extend(modules_to_fuse_layer)

        return modules_to_fuse_

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        """
        This function performs the following operations on an input tensor 'x':
        	- Convolution using conv1()
        	- Batch Normalization using bn1()
        	- ReLU activation using relu()
        	- Max pooling using maxpool()
        	- Three more convolutional layers (layer1(), layer2(), and layer3())
        followed by batch normalization (bn4()) and ReLU activation (relu4())
        	- Average pooling (avgpool())
        	- Flattening (torch.flatten())
        	- Fully connected layer (fc())
        It returns the output of the fully connected layer.

        Args:
            x (Tensor): In this TorchScript implementation of a Neural Network
                class `_forward_impl()` method; `x` is the input Tensor to be
                processed. It first passes through successive layers like `conv1()`,
                `bn1()`, `relu()`, `maxpool()` and 3 further layer modules
                (`layer1()`, `layer2()`, and `layer3()`). Later on it runs avg
                pool () on it , followed by flatten(()) , then the fully connected(fc())
                layer and return the output.
                In simple terms , the `x` input parameter serves as the input to
                the entire Neural Network and gets passed through all its layers
                sequentially for forward propagation

        Returns:
            Tensor: The output of the given function is a tensor.

        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        This function implements a neural network layer. It first quantizes the
        input tensor "x" using a custom implementation called _forward_impl(), and
        then applies some computation. Finally it de-quantizes the output and
        returns it.

        Args:
            x (Tensor): Here's what the "x" parameter does based on the given code
                snippet:
                
                	- The function takes one argument 'x', which is a Tensor object.
                	- It goes through a sequence of modifications before being returned
                at the end of the forward function
                	- These transformations involve calling methods named _forward_impl()
                and dequant()
                
                In brief , "x" acts as the input that is processed and transformed
                by the forward function before being returned .

        Returns:
            Tensor: Based on the provided function signature `def forward(self`',
            the output returned by this function would be `Tensor`.

        """
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x


def _resnet(
    block: Type[BasicBlock],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META = {
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}


class ResNet18_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet18-f37072fd.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 11689512,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 69.758,
                    "acc@5": 89.078,
                }
            },
            "_ops": 1.814,
            "_file_size": 44.661,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


@handle_legacy_interface(weights=("pretrained", ResNet18_Weights.IMAGENET1K_V1))
def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    weights = ResNet18_Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)