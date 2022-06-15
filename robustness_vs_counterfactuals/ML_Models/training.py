import torch
import copy

import os
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from ML_Models.LR.model import Regression
import ML_Models.ANN.model as model_ann
import ML_Models.data_loader as loader


def training(model, train_loader, test_loader, ml_model,
             dir_name, learning_rate, epochs, dataset, task,
             weighted_model: bool = False):
    
    loaders = {'train': train_loader,
               'test': test_loader}
    
    # model collector
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    best_loss = 1000000
    
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Declaring optimizer and loss
    if task == "regression":
        criterion = nn.MSELoss(reduce=False)
    else:
        criterion = nn.BCELoss(reduce=False)
    
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    
    # training
    for e in range(epochs):
        if e % 5 == 0:
            print('Epoch {}/{}'.format(e, epochs - 1))
            print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluation mode
            
            running_loss = 0.0
            running_acc = 0.0
            running_f1 = 0.0
            
            for i, (inputs, labels, indeces) in enumerate(loaders[phase]):
                
                inputs = inputs.to(device)
                labels = labels.to(device).type(torch.long)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    y_pred = model(inputs.float()).reshape(-1)
                    batch_weights = model.data_weights_vector[indeces]
                    loss = torch.mean(torch.multiply(batch_weights, criterion(y_pred.float(), labels.float())))
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistics
                preds = y_pred.data >= 0.5
                
                if task == 'classification':
                    running_acc += accuracy_score(labels.numpy(), preds.view(-1).long().numpy())
                    running_loss += loss.item()  # * inputs.size(0)
                    running_f1 += f1_score(labels.numpy(), preds.view(-1).long().numpy())
                else:
                    running_loss += (loss.item() / inputs.size(0))

            if task == 'classification':
                epoch_loss = running_loss / (i + 1)
                epoch_acc = running_acc / (i + 1)
                epoch_f1 = running_f1 / (i + 1)
            else:
                epoch_loss = running_loss / (i + 1)
            
            if e % 5 == 0:
                if task == 'classification':
                    print(f'{phase}: Loss: {epoch_loss:.4f} | F1-score: {epoch_f1:.4f} | Accuracy: {epoch_acc:.4f}')
                else:
                    print(f'{phase}: MSE: {epoch_loss:.4f}')
            
            # deep copy the model
            if task == 'classification':
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            else:
                if phase == 'test' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
    
    if weighted_model:
        # save model which includes trainable weight parameter option
        # currently the weights are all set to 1!
        # and during model training all weights are non-trainable
        torch.save(model.state_dict(best_model_wts),
                   './ML_Models/Saved_Models/{}/weighted_{}_{}{}.pt'.format(dir_name, dataset,
                                                                   ml_model, learning_rate))
    else:
        # save vanilla model
        torch.save(model.state_dict(best_model_wts),
                   './ML_Models/Saved_Models/{}/{}_{}{}.pt'.format(dir_name, dataset,
                                                                   ml_model, learning_rate))


def main(ml_model: str = 'ann', weighted_model=True, width: int = 100,
         names: list = ['compas', 'adult', 'folktables-c'],
         training_flag: bool = True):
    
    current_dir = os.getcwd()
    
    if ml_model == 'lr':
        width = 0
        dir_name = 'LR'
    elif ml_model == 'ann':
        width = width
        dir_name = 'ANN'
        
    compas_dict = {
        "path": './Data_Sets/COMPAS/',
        "filename_train": 'compas-train.csv',
        "filename_test": 'compas-test.csv',
        "label": "risk",
        "task": "classification",
        "batch_size": 32,
        "lr": 0.002,
        'epochs': 50
    }
    
    adult_dict = {
        "path": "./Data_Sets/Adult/",
        "filename_train": 'adult-train.csv',
        "filename_test": 'adult-test.csv',
        "label": 'income',
        "task": "classification",
        "batch_size": 256,
        "lr": 0.002,
        'epochs': 50
    }
    
    folktables_c_dict = {
        "path": "./Data_Sets/Folktables/",
        "filename_train": "folktables_classification-train.csv",
        "filename_test": "folktables_classification-test.csv",
        "label": ">50K",
        "task": "classification",
        "batch_size": 64,
        "lr": 0.002,
        'epochs': 50
    }
    
    data_meta_dictionaries = {
        "compas": compas_dict,
        "adult": adult_dict,
        "folktables-c": folktables_c_dict,
    }
    
    
    for name in names:
        
        data_meta_info = data_meta_dictionaries[name]
        print('-------------------------------------')
        print('Data set:', name)
        fname = f'{current_dir}/ML_Models/Saved_Models/{dir_name}/{name}_width{width}_lr{data_meta_info["lr"]}.pt'
        
        if not training_flag:
            if os.path.isfile(fname):
                continue
                
        dataset_train = loader.DataLoader_Tabular(path=data_meta_info["path"],
                                                  filename=data_meta_info["filename_train"],
                                                  label=data_meta_info["label"])
        
        dataset_test = loader.DataLoader_Tabular(path=data_meta_info["path"],
                                                 filename=data_meta_info["filename_test"],
                                                 label=data_meta_info["label"])
        
        input_size = dataset_train.get_number_of_features()
            
        # Define the model
        if ml_model == 'ann':
            model = model_ann.ANN(input_size, hidden_layer=width, num_of_classes=1,
                                  task=data_meta_info["task"], weighted_model=weighted_model,
                                  train_set_size=len(dataset_train))
        elif ml_model == 'lr':
            model = Regression(input_size, num_of_classes=1,
                               task=data_meta_info["task"], weighted_model=weighted_model,
                               train_set_size=len(dataset_train))
        else:
            print('Invalid model type')
            exit(0)
            
        trainloader = DataLoader(dataset_train, batch_size=data_meta_info["batch_size"], shuffle=True)
        testloader = DataLoader(dataset_test, batch_size=data_meta_info["batch_size"], shuffle=True)
        
        training(model, trainloader, testloader, ml_model, dir_name, data_meta_info["lr"],
                 data_meta_info["epochs"], name + '_width' + str(width),
                 task=data_meta_info["task"], weighted_model=weighted_model)
        
           
if __name__ == "__main__":
    # execute training
    main()
