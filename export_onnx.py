import torch
from trainer import Trainer
import sys
import torchvision as tv
from torchvision.models import resnet34

epoch = int(sys.argv[1])
model = resnet34(weights='DEFAULT')
model.fc = torch.nn.Sequential(torch.nn.Linear(512,2),torch.nn.Sigmoid()) 
crit = torch.nn.BCELoss()
trainer = Trainer(model, crit)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('/Users/sebh/Developer/Pytorch_Challenge/checkpoints/checkpoint_{:03d}.onnx'.format(epoch))
