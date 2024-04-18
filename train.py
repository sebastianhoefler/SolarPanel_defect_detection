import torch
torch.manual_seed(42)

from data import ChallengeDataset
from trainer import Trainer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet34, resnet50, ResNet50_Weights
import warnings
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
import os
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder

from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_warmup as warmup
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


mps_device = torch.device("mps")



# load the data from the csv file and 
data = pd.read_csv('/Users/sebh/Developer/Pytorch_Challenge/src_to_implement/data.csv', sep=';')

data['class'] = data['crack'].astype(str) + data['inactive'].astype(str)

# Encode class labels to integers
le = LabelEncoder()
data['class_encoded'] = le.fit_transform(data['class'])

# Separate features and target
X = data[['filename']]
y = data['class_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform oversampling on the training data
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

# After resampling, you can split the 'class_encoded' back into the original columns if needed
y_train_res_df = pd.DataFrame(y_train_res, columns=['class_encoded'])
y_train_res_df['class'] = le.inverse_transform(y_train_res_df['class_encoded'])
y_train_res_df[['crack', 'inactive']] = pd.DataFrame(y_train_res_df['class'].str.extract('(\d)(\d)').values.tolist(), dtype=int)
train_set = pd.concat([X_train_res, y_train_res_df[['crack', 'inactive']]], axis=1)

# Transform y_test back into the original columns
y_test_df = pd.DataFrame(y_test, columns=['class_encoded'])
y_test_df['class'] = le.inverse_transform(y_test_df['class_encoded'])
y_test_df[['crack', 'inactive']] = y_test_df['class'].str.split("", n=2, expand=True)[[1,2]].astype(int)
val_set = pd.concat([X_test, y_test_df[['crack', 'inactive']]], axis=1)

# Set up data loading for the training and validation set each using DataLoader and ChallengeDataset objects
train_data = DataLoader(ChallengeDataset(train_set, 'train'), batch_size=16, shuffle=True)
test_data = DataLoader(ChallengeDataset(val_set, 'val'), batch_size=16, shuffle=True)

# create an instance of our ResNet model
model = resnet34(weights='DEFAULT') #pretrained. Maybe try IMAGENET1k_V2 later?

# make sure the parameters are trainable
for param in model.parameters():
    param.requires_grad = True

model.fc = torch.nn.Sequential(torch.nn.Linear(512,2),
                               torch.nn.Sigmoid())


loss_fn = torch.nn.BCELoss()

#set up the optimizer
optimizer = torch.optim.SGD(model.parameters(), 
                            lr=0.0025, 
                            momentum=0.9,
                            nesterov=False,
                            weight_decay=0.0001)

# scheduler
scheduler = MultiStepLR(optimizer, 
                        milestones=[2, 10, 15, 17, 20], 
                        gamma=0.3) #changed from 0.3
scheduler2 = MultiStepLR(optimizer, 
                        milestones=[5,70], 
                        gamma=0.5)


# create an object of type Trainer and set its early stopping criterion
trainer = Trainer(model, loss_fn, optimizer, [scheduler, scheduler2], train_data, test_data, True, 20)
# TODO

# go, go, go... call fit on trainer
res = trainer.fit(epochs = 50)


