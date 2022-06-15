import numpy as np
import torch
import torch.nn as nn
import os

from torch.utils.data import DataLoader
import ML_Models.data_loader as loader
import Recourse_Methods.Generative_Model.model as model_vae


def training(model, train_loader, test_loader, learning_rate, epochs, batch_size, lambda_reg, data_name):
    
    loaders = {'train': train_loader,
               'test': test_loader}

    optimizer_model = torch.optim.Adam(model.parameters(),
                                       lr=learning_rate,
                                       weight_decay=lambda_reg)
    
    # model collector
    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0
    # best_loss = 1000000
    
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Train the VAE with the new prior
    ELBO_train = np.zeros((epochs, 1))
    ELBO_test = np.zeros((epochs, 1))
    
    for epoch in range(epochs):
        
        if epoch % 5 == 0:
            print('-' * 10)
            print('Epoch {}/{}'.format(epoch, epochs - 1))
    
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluation mode
    
            # Initialize the losses
            train_loss = 0
            test_loss = 0
    
            # Train for all the batches
            for batch_idx, (data, _, _) in enumerate(loaders[phase]):
                data = data.view(data.shape[0], -1).float()
                
                optimizer_model.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    MU_X_eval, LOG_VAR_X_eval, Z_ENC_eval, MU_Z_eval, LOG_VAR_Z_eval = model(data)
        
                    # The VAE loss
                    loss = model.VAE_loss(x=data, mu_x=MU_X_eval, log_var_x=LOG_VAR_X_eval,
                                          mu_z=MU_Z_eval, log_var_z=LOG_VAR_Z_eval)
        
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer_model.step()
                        train_loss += loss.detach().item() / batch_size
                    else:
                        test_loss += loss.detach().item() / batch_size
                
            if epoch % 10 == 0 and phase == 'train':
                ELBO_train[epoch] = train_loss
                print("[Epoch: {}| {}/{}] [ELBO: {:.3f}]".format(phase, epoch, epochs, ELBO_train[epoch, 0]))
            elif epoch % 10 == 0 and phase == 'test':
                ELBO_test[epoch] = test_loss
                print("[Epoch: {}| {}/{}] [ELBO: {:.3f}]".format(phase, epoch, epochs, ELBO_test[epoch, 0]))

    torch.save(model.state_dict(),
               '../../Recourse_Methods/Generative_Model/Saved_Models/vae_{}.pt'.format(data_name))

    print("Training on " + data_name + " completed")


def main(names: list = ['compas', 'adult', 'folktables-c'],
         training_flag: bool = False):
    names = ["german"]
    current_dir = os.getcwd()

 
    # Folktables
    folktables_c_dict = {
        "path": "./Data_Sets/Folktables/",
        "filename_train": "folktables_classification-train.csv",
        "filename_test": "folktables_classification-test.csv",
        "label": ">50K",
        "batch_size": 256,
        "lr": 0.002,
        "epochs": 50,
        "d": 6,
        "H1": 25,
        "H2": 25,
        "activFun": nn.Softplus(),
        'lambda_reg': 1e-6
    }

    # Adult
    adult_dict = {
        "path": "./Data_Sets/Adult/",
        "filename_train": 'adult-train.csv',
        "filename_test": 'adult-test.csv',
        "label": 'income',
        "task": "classification",
        "batch_size": 256,
        "lr": 0.002,
        "epochs": 50,
        "d": 6,
        "H1": 25,
        "H2": 25,
        "activFun": nn.Softplus(),
        'lambda_reg': 1e-6
    }

    #German
    german_dict = {
        "path": '../../Data_Sets/German_Credit_Data/',
        "filename_train": 'german-train.csv',
        "filename_test": 'german-test.csv',
        "label": "credit-risk",
        "task": "classification",
        "lr": 1e-3,
        "d": 6,
        "H1": 10,
        "H2": 10,
        "activFun": nn.Softplus(),
        "batch_size": 32,
        'lambda_reg': 1e-6,
        "epochs": 50
    }
    
    # Compas
    compas_dict = {
        "path": "./Data_Sets/COMPAS/",
        "filename_train": 'compas-train.csv',
        "filename_test": 'compas-test.csv',
        "label": "risk",
        "task": "classification",
        "batch_size": 32,
        "lr": 0.002,
        "epochs": 50,
        "d": 6,
        "H1": 10,
        "H2": 10,
        "activFun": nn.Softplus(),
        'lambda_reg': 1e-6
    }
    
    data_meta_dictionaries = {
        "compas": compas_dict,
        "adult": adult_dict,
        "german": german_dict
    }
    
    for data_name in names:
    
        fname = f'../../Recourse_Methods/Generative_Model/Saved_Models/vae_{data_name}.pt'
        data_meta_info = data_meta_dictionaries[data_name]
        
        # Check if pretrained VAE already exists; if not, train VAE model
        if not training_flag:
            if os.path.isfile(fname):
                continue
    
        dataset_train = loader.DataLoader_Tabular(path=data_meta_info["path"],
                                                  filename=data_meta_info["filename_train"],
                                                  label=data_meta_info["label"])
    
        dataset_test = loader.DataLoader_Tabular(path=data_meta_info["path"],
                                                 filename=data_meta_info["filename_test"],
                                                 label=data_meta_info["label"])
    
        # initial = int(0.33 * epochs)
        
        # The model and the optimizer for the VAE
        input_size = dataset_train.get_number_of_features()
        model = model_vae.VAE_model(input_size,
                                   data_meta_info['activFun'],
                                   data_meta_info['d'],
                                   data_meta_info['H1'],
                                   data_meta_info['H2'])
    
        epochs = data_meta_info["epochs"]
        trainloader = DataLoader(dataset_train,
                                 batch_size=data_meta_info["batch_size"],
                                 shuffle=True)
        testloader = DataLoader(dataset_test,
                                batch_size=data_meta_info["batch_size"],
                                shuffle=True)

        training(model, trainloader, testloader,
                 data_meta_info["lr"],
                 data_meta_info["epochs"],
                 data_meta_info["batch_size"],
                 data_meta_info["lambda_reg"],
                 data_name)


if __name__ == "__main__":
    # run vae training
    main()
