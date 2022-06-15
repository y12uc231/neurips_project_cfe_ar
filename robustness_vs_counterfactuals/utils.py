# Utils
import os
import numpy as np
import pandas as pd
import torch

# Recourse Methods
 
from Recourse_Methods.manifold_methods import REVISE
from Recourse_Methods.manifold_methods import CCHVAE
from Recourse_Methods.ip_methods import AR


def _get_input_subset(model, inputs: torch.tensor,
                      subset_size: int = 100,
                      decision_threshold: float = 0.5) -> torch.tensor:
    
    """
    Get negatively classified inputs & return their predictions
    """
    
    yhat = (model(inputs) > decision_threshold) * 1
    print(yhat, model(inputs).shape)
    check = (model(inputs) < decision_threshold).detach().numpy()
    selected_indices = np.where(check)[0]
    print("selected_indices values : ",selected_indices)
    input_subset = inputs[selected_indices]
    predicted_label_subset = yhat[selected_indices]
    print("Input values : ", input_subset)
    return input_subset[0:subset_size, :], predicted_label_subset[0:subset_size]


def get_recourses(data_loader, ml_model, vae_model, dataname: str, threshold: float = 0.5,
                  traindata_flag: bool = True, how_to: str = 'load',
                  subset_size: int = 100,
                  recourse_names: list = ['scfe', 'cchvae', 'revise', 'ar']) -> dict:
    
    """
    :param how_to: str --> one of: {'load', 'overwrite', 'new'}
    :return:
    """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data_iter = iter(data_loader)
    inputs, labels, indeces = data_iter.next()
    inputs = inputs.to(device).float()
    
    inputs, predicted_classes = _get_input_subset(ml_model, inputs, subset_size=subset_size, decision_threshold=0.5)
    inputs = inputs.numpy()
    print(f" Computing recoureses for {inputs.shape[0]} individuals ...")
    
    # ---------------------------
    # INITIALIZE RECOURSE METHODS
    # ---------------------------
    # TODO: add hyperparameter file to keep track of HPs in one place
    # TODO: add causal recourse method!
    # TODO: Adjust AR setup: add train loader to get_recourses() & so we can load train data for AR
    
    # This method makes the "IMF assumption" and uses gradients in input space
    scfe = SCFE(classifier=ml_model, lr=1e-3, _lambda=0.00, step=0.00, max_iter=1000, target_threshold=threshold)
    # This method makes the "IMF assumption" and uses gradients in input space
    inps = pd.DataFrame(inputs)
    ar = AR(classifier=ml_model, train_data=inps)
    # This method makes the "Manifold assumption" and uses gradients in latent space
    revise = REVISE(classifier=ml_model, model_vae=vae_model, _lambda=0.00, step=0.05, lr=1e-2, max_iter=1000,
                    target_threshold=threshold)
    # This method makes the "Manifold assumption" and uses random search in latent space
    cchvae = CCHVAE(classifier=ml_model, model_vae=vae_model, step=0.01, max_iter=1000, target_threshold=threshold)
    
    # ---------------------------
    # COLLECT RECOURSE RESULTS
    # ---------------------------

    if traindata_flag:
        regime = 'train'
    else:
        regime = 'test'
    
    recourse_methods = {
        'scfe': scfe,
        'cchvae': cchvae,
        'revise': revise,
        'ar': ar
    }
    
    results_dict = {
        "scfe": 0,
        "cchvae": 0,
        "revise": 0,
        "ar": 0
    }
    
    current_dir = os.getcwd()
    
    for it, recourse_name in enumerate(recourse_names):
        
        file_check = os.path.isfile(
            f"{current_dir}/Recourse_Methods/Saved_Intermediate_Results/Distances_{dataname}_{regime}_{recourse_name}")
        
        if how_to == 'load' and file_check:
            # Load recourses if file exists
            
            print(f"  Loading recourses by: {recourse_names[it]}")
            
            counterfactuals = pd.read_csv(
                f"{current_dir}/Recourse_Methods/Saved_Intermediate_Results/CEs_{dataname}_{regime}_{recourse_name}")
            counterfactuals = counterfactuals.values
            
            distances = pd.read_csv(
                f"{current_dir}/Recourse_Methods/Saved_Intermediate_Results/Distances_{dataname}_{regime}_{recourse_name}")
            distances = distances.values
            
            inputs = pd.read_csv(
                f"{current_dir}/Recourse_Methods/Saved_Intermediate_Results/Inputs_{dataname}_{regime}_{recourse_name}")
            inputs = inputs.values


        elif how_to == 'overwrite' or how_to == 'new':
            
            print(f"  Finding recourses using: {recourse_names[it]}")
            
            # Get recourse results
            counterfactuals = np.zeros_like(inputs)
            distances = np.zeros_like(inputs[:, 0])
            for j in range(inputs.shape[0]):
                pred_class = predicted_classes[j]
                counterfactual, distance = recourse_methods[recourse_name].generate_counterfactuals(
                    query_instance=torch.tensor(inputs[j]).reshape(-1),
                    target_class=1-pred_class)
                counterfactuals[j, :] = counterfactual.detach().numpy()
                distances[j] = distance.detach().numpy()
                
            # Saving results to CSVs
            results_to_save_ces = pd.DataFrame(counterfactuals)
            results_to_save_ces.to_csv(
                f"{current_dir}/Recourse_Methods/Saved_Intermediate_Results/CEs_{dataname}_{regime}_{recourse_name}",
                                        index=False)
            
            results_to_save_distances = pd.DataFrame(distances)
            results_to_save_distances.to_csv(
                f"{current_dir}/Recourse_Methods/Saved_Intermediate_Results/Distances_{dataname}_{regime}_{recourse_name}",
                                              index=False)
            
            results_to_save_inputs = pd.DataFrame(inputs)
            results_to_save_inputs.to_csv(
                f"{current_dir}/Recourse_Methods/Saved_Intermediate_Results/Inputs_{dataname}_{regime}_{recourse_name}",
                                           index=False)
        else:
            print("Illegal setting: Either no saved intermediate results or wrong string")
            exit(0)
        
        results = {
            "recourses": torch.tensor(counterfactuals),
            "inputs": torch.tensor(inputs),
            "distances": torch.tensor(distances)
        }
        results_dict[recourse_name] = results
        
    return results_dict
