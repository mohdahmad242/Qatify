import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from torch.utils.data import DataLoader

from model.resnet18 import resnet18
from utils.getDataset import get_data
from utils.train import train
# from utils.evaluate import evaluate


from evaluate import evaluate

# Get the train and test loader
dataset_name = "cifar-100" # Supported cifar-10 | cifar-100
trainloader, testloader = get_data(dataset_name, 128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Current available device - {device}')
model = resnet18(pretrained=True)
model = model.to(device)
# print(model)

# Step 1: architecture changes
# QuantStubs (we will do FloatFunctionals later)
# Done

# Step 2: fuse modules (recommended but not necessary)
modules_to_list = model.modules_to_fuse()

# It will keep Batchnorm
model.eval()
# fused_model = torch.ao.quantization.fuse_modules_qat(model, modules_to_list)

# This will fuse BatchNorm weights into the preceding Conv
fused_model = torch.ao.quantization.fuse_modules(model, modules_to_list)

# Step 3: Assign qconfigs
from torch.ao.quantization.fake_quantize import FakeQuantize
activation_qconfig = FakeQuantize.with_args(
    observer=torch.ao.quantization.observer.HistogramObserver.with_args(
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
    )
)

weight_qconfig = FakeQuantize.with_args(
    observer=torch.ao.quantization.observer.PerChannelMinMaxObserver.with_args(
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
    )
)

qconfig = torch.quantization.QConfig(activation=activation_qconfig,
                                      weight=weight_qconfig)
fused_model.qconfig = qconfig

# Step 4: Prepare for fake-quant
fused_model.train()
fake_quant_model = torch.ao.quantization.prepare_qat(fused_model)



print("\nFloat")
evaluate(model, 'cpu')


print("\nFused Model")
evaluate(fused_model, 'cpu')


print("\nFake quant - PTQ")
evaluate(fake_quant_model, 'cpu')

fake_quant_model.apply(torch.ao.quantization.fake_quantize.disable_observer)

print("\nFake quant - post-PTQ")
evaluate(fake_quant_model, 'cpu')


torch.backends.quantized.engine = 'qnnpack'
# Step 5: convert (true int8 model)
converted_model = torch.ao.quantization.convert(fake_quant_model)

print("\nConverted model")
evaluate(converted_model, 'cpu')

# Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# train(model=model, trainloader=trainloader, criterion=criterion, optimizer=optimizer, device=device, epochs=10)

# # Evaluate the model
# accuracy, avg_inference_time = evaluate(model, testloader, device)
