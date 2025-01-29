import torch
from torchvision import models
from pytorch_nndict.apis import torch_quantizer
model=models.resnet18(pretrained=True)
quantizer=torch_quantizer('calib',model,(1,3,224,224))
quantized_model=quantizer.quant_model

print(quantized_model.eval())

quantizer.export_quant_config()
