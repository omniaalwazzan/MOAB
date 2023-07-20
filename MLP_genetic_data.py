import pandas as pd
import pickle
import os
from Pathomic_data_cleaning import getCleanAllDataset
import torch

import math
import numpy as np
import scipy
pd.options.display.max_rows = 999
import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
from sklearn.preprocessing import MinMaxScaler    
from torch.nn import init, Parameter
from torchsummary import summary
import torch.nn.functional as F
import seaborn as sns    
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


### Load PNAS Splits
dataroot = r'C:\Users\omnia\.spyder-py3\pathomic_csv\fold1/'
pnas_splits = pd.read_csv(dataroot+'pnas_splits.csv')
pnas_splits.columns = ['TCGA ID']+[str(k) for k in range(1, 16)]
pnas_splits.index = pnas_splits['TCGA ID']
pnas_splits = pnas_splits.drop(['TCGA ID'], axis=1)
#model='cox_omic'
model= 'cox_grade'
### Loads Data
ignore_missing_moltype = True if model in ['cox_omic', 'cox_moltype', 'cox_grade+moltype', 'all'] else False
ignore_missing_histype = True if model in ['cox_histype', 'cox_grade', 'cox_grade+moltype', 'all'] else False
all_dataset = getCleanAllDataset(dataroot=dataroot, ignore_missing_moltype=ignore_missing_moltype, 
                                 ignore_missing_histype=ignore_missing_histype)[1]


class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

class Linear_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Linear_ = nn.Sequential(
            nn.Linear(in_channels,out_channels),
            #nn.ReLU(inplace=True),
            nn.LayerNorm(out_channels)
            )

    def forward(self, x):
        return self.Linear_(x)
    
class MulticlassClassification_1(nn.Module):
    
    def __init__(self, num_class=3):
        super(MulticlassClassification_1, self).__init__()
        self.layer_1 = Linear_Layer(input_dim, 80)
        self.layer_2 = Linear_Layer(80, 40)
        self.layer_3 = Linear_Layer(40, 32)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_out = nn.Linear(32, num_class) 
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.dropout(x) 
        x = self.layer_3(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x
def model_gens_2() -> MulticlassClassification_1:
    model = MulticlassClassification_1()
    return model

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc



for k in pnas_splits.columns:
    pat_train = list(set(pnas_splits.index[pnas_splits[k] == 'Train']).intersection(all_dataset.index))
    pat_test = list(set(pnas_splits.index[pnas_splits[k] == 'Test']).intersection(all_dataset.index))
    feats = all_dataset.columns.drop(['Survival months', 'censored', 'Histology','Molecular subtype', 'Histomolecular subtype',])
    df_train = all_dataset.loc[pat_train]
    df_test = all_dataset.loc[pat_test]
    
    df_train = df_train[feats]
    df_train = df_train.set_index('TCGA ID')
    df_train_new = df_train.loc[:,df_train.columns!='Grade'] # move the grade to the last col
    df_train_new['Grade']=df_train['Grade'] # move the grade to the last col

    df_test = df_test[feats]
    #df_test = df_test[(df_test['Grade'] >= 0)] # this to remove the missing data like grade =-1
    df_test = df_test.set_index('TCGA ID')
    df_test_new = df_test.loc[:,df_test.columns!='Grade']# move the grade to the last col
    df_test_new['Grade']=df_test['Grade'] # move the grade to the last col
    
    X_train = df_train_new.iloc[:, 0:-1]
    y_train = df_train_new.iloc[:, -1]

    input_dim=len(X_train.columns)

    X_test = df_test_new.iloc[:, 0:-1]
    y_test = df_test_new.iloc[:, -1]


    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())


    train_loader = DataLoader(dataset=train_dataset,batch_size=8)
    test_loader = DataLoader(dataset=test_dataset, batch_size=8)
    
    
    model = model_gens_2()
    # summary(model, input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005) 



    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    EPOCHS = 10


    print("Begin training.")
    for e in tqdm(range(1, EPOCHS+1)):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            
            y_train_pred = model(X_train_batch)
            
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)
            
            train_loss.backward()
            optimizer.step()
            
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
            
            

                
                
                
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        #loss_stats['val'].append(val_epoch_loss/len(test_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        #accuracy_stats['val'].append(val_epoch_acc/len(test_loader))
        
                                  
        
        #print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(test_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(test_loader):.3f}')
        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}')

     
for k in pnas_splits.columns:
    pat_test = list(set(pnas_splits.index[pnas_splits[k] == 'Test']).intersection(all_dataset.index))
    feats = all_dataset.columns.drop(['Survival months', 'censored','Histology','Molecular subtype', 'Histomolecular subtype'])
    df_test = all_dataset.loc[pat_test]
    
    df_test = df_test[feats]
    df_test = df_test[(df_test['Grade'] >= 0)] # this to remove the missing data like grade =-1
    df_test = df_test.set_index('TCGA ID')
    df_test_new = df_test.loc[:,df_test.columns!='Grade']
    df_test_new['Grade']=df_test['Grade']
    
    X_train = df_train_new.iloc[:, 0:-1]

    input_dim=len(X_train.columns)

    X_test = df_test_new.iloc[:, 0:-1]
    y_test = df_test_new.iloc[:, -1]


    X_test, y_test = np.array(X_test), np.array(y_test)
    test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())


    
    test_loader = DataLoader(dataset=test_dataset, batch_size=6)
    
    truelabels = []
    predictions = []
    proba = []
    pre = []
    
    with torch.no_grad():
        model.eval()
        for data, target in test_loader:
            data = data.to(device=device,dtype=torch.float)
            target = target.to(device=device,dtype=torch.float)
            truelabels.extend(target.cpu().numpy())

            output = model(data)
            probs = F.softmax(output, dim=1)[:, 1]# assuming logits has the shape [batch_size, nb_classes]
            #top_p, top_class = prob.topk(1, dim = 1)
            preds = torch.argmax(output, dim=1) # this to plot the confusion matrix
            _, predicted = torch.max(output.data, 1)
            predictions.extend(predicted.cpu().numpy())
            proba.extend(probs.cpu().numpy())
            pre.extend(preds.cpu().numpy())
            from sklearn.metrics import f1_score
        print('F1-score micro for MLP classifer:')
        print(f1_score(truelabels, predictions, average='micro'))
        print(classification_report(truelabels, predictions))
            
        cm = confusion_matrix(truelabels, pre)
        classes= ['GradeII', 'GradeIII', 'GradeIV']
        #classes= ['0']
        tick_marks = np.arange(len(classes))
        
        df_cm = pd.DataFrame(cm, index = classes, columns = classes)
        plt.figure(figsize = (7,7))
        sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
        plt.xlabel("Predicted label", fontsize = 20)
        plt.ylabel("Ground Truth", fontsize = 20)
        plt.show()

    


 
