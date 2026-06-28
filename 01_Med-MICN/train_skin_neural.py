
import os
import torch
import torch_explain as te
from torch_explain.nn.concepts import ConceptReasoningLayer
from sklearn.metrics import accuracy_score
from torch_explain.nn.concepts import ConceptReasoningLayer
import torch.nn.functional as F
import argparse
from torchvision.models import resnet50,densenet169,vgg16
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from models import Neural_Concat_Model
from torch.utils.data import random_split
from dataset.ddi_concept_dataset import DDI_Dataset
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
parser = argparse.ArgumentParser(description='Settings for creating model')

parser.add_argument("--image_dir",type=str,default='Your Image Folder Path')
parser.add_argument("--embedding_size",type=int,default=8)
parser.add_argument("--batchsize",type=int,default=32)
parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--test_epoch",type=int,default=1)
parser.add_argument("--backbone",type=str,default='RN50')
parser.add_argument("--neural_explain",type=bool,default=False)

args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)

concepts=['Vesicle','Papule','Exudate','Fissure','Xerosis','Warty/Papillomatous','Brown(Hyperpigmentation)','Translucent','White(Hypopigmentation)','Erythema','Wheal','Pigmented','Cyst']
concepts_num=len(concepts)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

dataset = DDI_Dataset(root=args.image_dir,
                    transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
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

test_size = total_size - train_size 
train_set, test_set = random_split(dataset, [train_size, test_size])
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.batchsize, shuffle=False, num_workers=0, pin_memory=device)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=args.batchsize, shuffle=False, num_workers=0, pin_memory=device)

if args.backbone=='RN50':
    feature_size=1000
    backbone = resnet50(pretrained=True)
    concept_encoder = torch.nn.Sequential(
    torch.nn.Linear(feature_size, 10),
    torch.nn.LeakyReLU(),
    te.nn.ConceptEmbedding(10, concepts_num, args.embedding_size),
    )
    task_predictor = torch.nn.Sequential(
        torch.nn.Linear(concepts_num*args.embedding_size+feature_size, feature_size),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(feature_size, 2),
        nn.Sigmoid()
    )
    task_concept_predictor=torch.nn.Sequential(
        torch.nn.Linear(concepts_num*args.embedding_size, 2),
        nn.Sigmoid()

    )
    task_neural_predictor = ConceptReasoningLayer(args.embedding_size, 2).to(device)
    model = Neural_Concat_Model(backbone,concept_encoder, task_neural_predictor,task_predictor,task_concept_predictor).to(device)

    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
if args.backbone=='vgg':
    feature_size=1000
    backbone = vgg16(pretrained=True)

    concept_encoder = torch.nn.Sequential(
    torch.nn.Linear(feature_size, 10),
    torch.nn.LeakyReLU(),
    te.nn.ConceptEmbedding(10, concepts_num, args.embedding_size),
    )
    task_predictor = torch.nn.Sequential(
        torch.nn.Linear(concepts_num*args.embedding_size+feature_size, feature_size),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(feature_size, 2),
        nn.Sigmoid()
    )
    task_concept_predictor=torch.nn.Sequential(
        torch.nn.Linear(concepts_num*args.embedding_size, 2),
        nn.Sigmoid()

    )
    task_neural_predictor = ConceptReasoningLayer(args.embedding_size, 2).to(device)
    model = Neural_Concat_Model(backbone,concept_encoder, task_neural_predictor,task_predictor,task_concept_predictor).to(device)

    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr= 5e-6)

if args.backbone=='DenseNet':
    feature_size = 1000
    backbone = densenet169(pretrained=True)
    task_concept_predictor=torch.nn.Sequential(
        torch.nn.Linear(concepts_num*args.embedding_size, 2),
        nn.Sigmoid()

    )
    concept_encoder = torch.nn.Sequential(
    torch.nn.Linear(feature_size, 128),
    torch.nn.LeakyReLU(),
    te.nn.ConceptEmbedding(128, concepts_num, args.embedding_size),
    )
    task_neural_predictor = ConceptReasoningLayer(args.embedding_size, 2).to(device)
    task_predictor = torch.nn.Sequential(
        torch.nn.Linear(concepts_num*args.embedding_size+feature_size, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 2),
        nn.Sigmoid()
    )
    model = Neural_Concat_Model(backbone,concept_encoder, task_neural_predictor,task_predictor,task_concept_predictor).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

if args.backbone=='DINOv2':
    # DINOv2 backbone configuration
    backbone = DINOv2Backbone(model_name="facebook/dinov2-base", freeze_backbone=False)
    feature_size = backbone.feature_dim  # feature dim from backbone
    
    task_concept_predictor = torch.nn.Sequential(
        torch.nn.Linear(concepts_num*args.embedding_size, 2),
        nn.Sigmoid()
    )
    
    concept_encoder = torch.nn.Sequential(
        torch.nn.Linear(feature_size, 256),
        torch.nn.LeakyReLU(),
        torch.nn.Dropout(0.1),
        te.nn.ConceptEmbedding(256, concepts_num, args.embedding_size),
    )
    
    task_neural_predictor = ConceptReasoningLayer(args.embedding_size, 2).to(device)
    
    task_predictor = torch.nn.Sequential(
        torch.nn.Linear(concepts_num*args.embedding_size+feature_size, 512),
        torch.nn.LeakyReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(512, 128),
        torch.nn.LeakyReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(128, 2),
        nn.Sigmoid()
    )
    
    model = Neural_Concat_Model(backbone, concept_encoder, task_neural_predictor, task_predictor, task_concept_predictor).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)



loss_from_concept_pred=torch.nn.CrossEntropyLoss()
loss_form_c = torch.nn.BCELoss()
loss_form_y = torch.nn.CrossEntropyLoss()
loss_form_neural = torch.nn.BCELoss()
model.train()
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
        optimizer.zero_grad() 
        path, images, labels,concepts_batch=batch
        bs=len(path)
            
        concepts_batch=torch.stack(concepts_batch,dim=1).to(device)
        concepts_batch=concepts_batch.float()
        feature,y_pred,y_pred_neural, c_emb, c_pred,y_pred_concept =model(images.to(device))

        # compute loss

        concept_loss =loss_form_c(c_pred, concepts_batch)
        concept_pred_loss=loss_from_concept_pred(y_pred_concept, (labels.long().ravel().to(device)))

        task_loss = loss_form_y(y_pred, (labels.long().ravel().to(device)))
        neural_loss = loss_form_neural(y_pred_neural,F.one_hot(labels.long().ravel()).float().to(device))
        loss = 0.1*concept_loss + 1.0*task_loss+0.1*neural_loss

        train_y_true = torch.cat((train_y_true, F.one_hot(labels.long().ravel()).float().to(device)))
        train_y_pred = torch.cat((train_y_pred, (y_pred.cpu() > 0.5).to(device)), dim=0)
       
        loss.backward()
        
        optimizer.step()
    train_task_accuracy = accuracy_score(train_y_true.cpu().numpy(), train_y_pred.cpu().numpy())

    local_explanations = task_neural_predictor.explain(c_emb, c_pred, 'local')
    global_explanations = task_neural_predictor.explain(c_emb, c_pred, 'global')
    

    all_y_true = []
    all_y_pred = []
    concept_true=[]
    all_y_concept_pred=[]
    concept_num_total=[]
    if epoch % args.test_epoch==0 :
        with torch.no_grad():  
            for batch in tqdm(test_dataloader, desc='test', disable=True):
                path, images, labels,concepts_batch=batch
                optimizer.zero_grad()      
                concepts_batch=torch.stack(concepts_batch,dim=1).to(device)
                concepts_batch=concepts_batch.float()
                feature,y_pred,y_pred_neural, c_emb, c_pred,c_concept_pred =model(images.to(device))

                local_explanations = task_neural_predictor.explain(c_emb, c_pred, 'local')

                global_explanations = task_neural_predictor.explain(c_emb, c_pred, 'global')
                if args.neural_explain==True:
                    print(local_explanations,global_explanations)

                all_y_true.extend(F.one_hot(labels.long().ravel()).float().cpu().numpy())
                all_y_pred.extend(y_pred.cpu().numpy() > 0.5)
                concept_true.append(sum(sum(concepts_batch.cpu().numpy()==(c_pred.cpu().numpy() > 0.5))))
                concept_num_total.append(concepts_batch.shape[0]*concepts_batch.shape[1])
                all_y_concept_pred.extend(c_concept_pred.cpu().numpy() > 0.5)
        model.train()

        task_accuracy = accuracy_score(np.array(all_y_true), np.array(all_y_pred))
        concept_accuracy = sum(concept_true)/sum(concept_num_total)
        y_concept_pred=accuracy_score(np.array(all_y_true), np.array(all_y_concept_pred))
        
        

        task_f1 = f1_score(np.array(all_y_true), np.array(all_y_pred), average='macro', zero_division=0)
        task_precision = precision_score(np.array(all_y_true), np.array(all_y_pred), average='macro', zero_division=0)
        task_recall = recall_score(np.array(all_y_true), np.array(all_y_pred), average='macro', zero_division=0)
        task_auc = roc_auc_score(np.array(all_y_true), np.array(all_y_pred))

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







