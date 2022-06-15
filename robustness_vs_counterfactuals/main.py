# Utils
import os
import torch
import torch.nn as nn
import numpy as np
from utils import get_recourses, get_performance_measures

# Models
import ML_Models.LR.model as model_lin
import ML_Models.ANN.model as model_ann
import Recourse_Methods.Generative_Model.model as model_vae

# Data
import ML_Models.data_loader as loader
from torch.utils.data import DataLoader

# Seeds
import random
torch.manual_seed(12345)
random.seed(12345)


def _correct_for_no_ces(array):
    indeces = np.where(~np.isnan(array))[0]
    if len(indeces) == 0:
        raise ValueError("Could not find any CEs")
    else:
        nCEs = array[indeces].shape[0]
        print(f"    Identified {nCEs} out of {array.shape[0]} counterfactual explanations")
        return array[indeces]


def main(func_class: str = 'ann', width: int = 100,
         datanames: list = ['adult', 'compas', 'folktables-c'],
         recourse_model_names : list = ['scfe', 'cchvae', 'revise', 'ar'],
         n_starting_instances: int = 1200, subset_size: int = 1000,
         how_to_get_recourse: str = 'load'):
    
    """
    how_to_get_recourse: str --> one of {'load', 'overwrite', 'new'}
    """
    
    current_dir = os.getcwd()
    # obtain generative model path
    vae_path = current_dir + "/Recourse_Methods/Generative_Model/Saved_Models/"
    
    # obtain classification model path
    if func_class == 'lr':
        width = 0
        m_path = current_dir + "/ML_Models/Saved_Models/LR/"
    elif func_class == 'ann':
        width = width
        m_path = current_dir + "/ML_Models/Saved_Models/ANN/"
    else:
        raise ValueError("Illegal setting. Only LR and ANN are supported thus far.")
    
    if subset_size >= n_starting_instances:
        raise ValueError("Illegal setting")
    
    compas_dict = {
        "data_path": './Data_Sets/COMPAS/',
        "filename_train": 'compas-train.csv',
        "filename_test": 'compas-test.csv',
        "label": "risk",
        "task": "classification",
        "lr": 1e-3,
        "d": 6,
        "H1": 10,
        "H2": 10,
        "activFun": nn.Softplus(),
        "n_starting_instances": n_starting_instances,
    }
    
    adult_dict = {
        "data_path": "./Data_Sets/Adult/",
        "filename_train": 'adult-train.csv',
        "filename_test": 'adult-test.csv',
        "label": 'income',
        "task": "classification",
        "lr": 1e-3,
        "d": 6,
        "H1": 25,
        "H2": 25,
        "activFun": nn.Softplus(),
        "n_starting_instances": n_starting_instances
    }
    
    folktables_c_dict = {
        "data_path": "./Data_Sets/Folktables/",
        "filename_train": "folktables_classification-train.csv",
        "filename_test": "folktables_classification-test.csv",
        "label": ">50K",
        "task": "classification",
        "lr": 1e-3,
        "d": 6,
        "H1": 25,
        "H2": 25,
        "activFun": nn.Softplus(),
        "n_starting_instances": n_starting_instances
    }
    
    data_meta_dictionaries = {
        "compas": compas_dict,
        "adult": adult_dict,
        "folktables-c": folktables_c_dict,
    }

    # Result container
    results_dict = dict()
    
    for dataname in datanames:
        
        data_meta_info = data_meta_dictionaries[dataname]
        
        # Add model path to dictionary
        data_meta_info["classifier_path"] = m_path + f"weighted_{dataname}_width{width}_{func_class}0.002.pt"
        data_meta_info["vae_path"] = vae_path + f"vae_{dataname}.pt"

        print('-------------------------------------')
        print('Data set:', dataname)
        
        dataset_test = loader.DataLoader_Tabular(path=data_meta_info["data_path"],
                                                 filename=data_meta_info["filename_test"],
                                                 label=data_meta_info["label"])
        
        dataset_train = loader.DataLoader_Tabular(path=data_meta_info["data_path"],
                                                  filename=data_meta_info["filename_train"],
                                                  label=data_meta_info["label"])
        
        input_size = dataset_test.get_number_of_features()
        number_train_samples = dataset_train.get_number_of_instances()
        model = None
        
        # Load the classifiers
        if func_class == 'ann':
            model = model_ann.ANN(input_size, hidden_layer=width, num_of_classes=1,
                                  train_set_size=number_train_samples)
            model.load_state_dict(torch.load(data_meta_info["classifier_path"], map_location=torch.device('cpu')))
        elif func_class == 'lr':
            model = model_lin.Regression(input_size, num_of_classes=1,
                                         train_set_size=number_train_samples)
            model.load_state_dict(torch.load(data_meta_info["classifier_path"], map_location=torch.device('cpu')))
        else:
            print('Invalid model type')
            exit(0)
            
        # Load the VAEs
        vae_model = model_vae.VAE_model(input_size,
                                        data_meta_info['activFun'],
                                        data_meta_info['d'],
                                        data_meta_info['H1'],
                                        data_meta_info['H2'])
        vae_model.load_state_dict(torch.load(data_meta_info["vae_path"]))
        
        testloader = DataLoader(dataset_test,
                                batch_size=data_meta_info["n_starting_instances"],
                                shuffle=True)
        
        trainloader = DataLoader(dataset_train,
                                 batch_size=data_meta_info["n_starting_instances"],
                                 shuffle=True)

        results_train = dict()
        results_test = dict()

        # get raw results
        results_train[dataname] = get_recourses(trainloader, ml_model=model, vae_model=vae_model, dataname=dataname,
                                                subset_size=subset_size, recourse_names=recourse_model_names,
                                                traindata_flag=True, how_to=how_to_get_recourse)

        results_test[dataname] = get_recourses(testloader, ml_model=model, vae_model=vae_model, dataname=dataname,
                                               subset_size=subset_size, recourse_names=recourse_model_names,
                                               traindata_flag=False, how_to=how_to_get_recourse)

        # package results
        intermeditat_dict = {
            'scfe': dict(),
            'cchvae': dict(),
            'revise': dict()
        }

        for recourse_model in recourse_model_names:
            dis_train = _correct_for_no_ces(results_train[dataname][recourse_model]['distances'].detach().numpy())
            dis_test = _correct_for_no_ces(results_test[dataname][recourse_model]['distances'].detach().numpy())
        
            # Collect the results in one dictionary
            intermeditat_dict[recourse_model]['train_distances'] = np.array(dis_train).reshape(-1)
            intermeditat_dict[recourse_model]['test_distances'] = np.array(dis_test).reshape(-1)
            intermeditat_dict[recourse_model]['all_distances'] = np.array([dis_train, dis_test]).reshape(-1)

        # Collect them all
        results_dict[dataname] = intermeditat_dict
            

if __name__ == "__main__":
    # execute experiments
    main()
