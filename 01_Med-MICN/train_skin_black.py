import os
import torch
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50, densenet169, vgg16
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch.nn as nn
from tqdm import tqdm, trange
import numpy as np
import wandb
from models import ResNetModel, DenseNetModel
from dataset.ddi_concept_dataset import DDI_Dataset
from datetime import datetime

parser = argparse.ArgumentParser(description='Settings for creating model')
parser.add_argument("--image_dir",type=str,default='Your Image Path')  
parser.add_argument("--embedding_size",type=int,default=8)
parser.add_argument("--batchsize",type=int,default=256)
parser.add_argument("--n_classes",type=int,default=2)
parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--test_epoch_per",type=int,default=1)
parser.add_argument("--feature_size",type=int,default=1000)
parser.add_argument("--backbone",type=str,default='RN50')


args = parser.parse_args()

num_classes = args.n_classes


device='cuda'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set random seed for reproducibility
torch.manual_seed(42)

concepts=['Vesicle','Papule','Macule','Plaque','Abscess','Pustule','Bulla','Patch','Nodule','Ulcer',
        'Crust','Erosion','Excoriation','Atrophy','Exudate','Purpura/Petechiae','Fissure','Induration'
]
concepts_num=len(concepts)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])


dataset = DDI_Dataset(root=args.image_dir,
                    transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        # AddGaussianNoise(mean=mean_gauss, stddev=stddev_gauss),
                        transforms.Lambda(lambda tensor: tensor.float() + 1e-6 if tensor.std() == 0 else tensor),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])]),
                        concepts=concepts)

dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32, shuffle=False,
            num_workers=0, pin_memory=device)
total_size = len(dataset)

train_size = int(0.85 * total_size)
test_size = total_size-train_size
train_set, test_set = random_split(dataset, [train_size, test_size])


train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=False, num_workers=8, pin_memory=device)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=8, pin_memory=device)

if args.backbone=='RN50':
    backbone = resnet50(pretrained=True)

    task_predictor = torch.nn.Sequential(
        torch.nn.Linear(1000, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, num_classes),
        nn.Sigmoid()
    )
    model=ResNetModel(backbone,task_predictor)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
if args.backbone=='vgg':
    backbone = vgg16(pretrained=True)
    task_predictor = torch.nn.Sequential(
        torch.nn.Linear(1000, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, num_classes),
        nn.Sigmoid()
    )
    model=torch.nn.Sequential(
    backbone,
    task_predictor 
    )
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
if args.backbone=='DenseNet':
    feature_size = 1000
    backbone = densenet169(pretrained=True)
    
    task_predictor = torch.nn.Sequential(
        torch.nn.Linear(1000, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, num_classes),
        nn.Sigmoid()
    )
    model=torch.nn.Sequential(
    backbone,
    task_predictor 
    )
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

loss_form_y = torch.nn.CrossEntropyLoss()


best_test_accuracy = 0.0
best_f1 = 0.0
def lambda_rule(epoch):
    return 0.1 / (1 + epoch)



for epoch in trange(args.epochs):
    train_y_true = torch.tensor([]).to(device)
    train_y_pred = torch.tensor([]).to(device)
    all_y_concept_pred=[]
    concept_num_total=[]
    
    for batch in train_dataloader:
        model.train()
        optimizer.zero_grad() 
        path, images, labels,concepts_batch=batch
        bs=len(path)
        y_pred=model(images.to(device))    
        loss = loss_form_y(y_pred, labels.to(device))

        # Predicted class
        _, predicted_classes = torch.max(y_pred, 1)
        train_y_true = torch.cat((train_y_true, F.one_hot(labels.long().ravel(), num_classes).float().to(device)))
        train_y_pred = torch.cat((train_y_pred, F.one_hot(predicted_classes.long().ravel(), num_classes).to(device)))
        loss.backward()
        
        optimizer.step()
    train_task_accuracy = accuracy_score(train_y_true.cpu().numpy(), train_y_pred.cpu().numpy())

    


    if epoch % args.test_epoch_per == 0:
        model.eval()
        all_y_true = []
        all_y_pred = []
        with torch.no_grad():  # No gradient during evaluation
            for batch in tqdm(test_dataloader, desc='test', disable=True):
                path, images, labels, concepts_batch = batch
                y_pred = model(images.to(device))

                # Predicted class
                _, predicted_classes = torch.max(y_pred, 1)

                all_y_true.extend(labels.cpu().numpy())
                all_y_pred.extend(predicted_classes.cpu().numpy())


        # Overall accuracy
        task_accuracy = accuracy_score(all_y_true, all_y_pred)

        # F1 score
        task_f1 = f1_score(all_y_true, all_y_pred, average='macro')

        # Precision
        task_precision = precision_score(all_y_true, all_y_pred, average='macro')

        # Recall
        task_recall = recall_score(all_y_true, all_y_pred, average='macro')

        # ROC AUC
        task_auc = roc_auc_score(all_y_true, all_y_pred, average='macro')

        if task_accuracy > best_test_accuracy:
            best_f1 = task_f1
            best_test_accuracy = task_accuracy

            print(  f"best_test_accuracy:{best_test_accuracy:.3g} "
                    f"best_f1:{best_f1:.3g} "
                    f"task_accuracy:{task_accuracy:.3g} "
                    f"task_f1:{task_f1:.3g} "
                    f"task_precision:{task_precision:.3g} "
                    f"task_recall:{task_recall:.3g} "
                    f"task_auc:{task_auc:.3g}"
                )





