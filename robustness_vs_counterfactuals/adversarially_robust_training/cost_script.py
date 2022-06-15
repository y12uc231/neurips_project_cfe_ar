#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Step 1 : Train baseline and robust models (linear and neural network) 
# Step 2 : Compute counterfactuals for both the models. 
# Step 3 : Comparison study between the two.


# In[ ]:





# In[2]:


# import recourse
# from recourse.cplex_helper import DEFAULT_CPLEX_PARAMETERS


# In[3]:


# Fixes :-
# Save dict of weights 
# 


# In[1]:


import sys
sys.path.append("/Users/skrishna/Documents/phd_codes/neurips_paper/robustness_vs_counterfactuals")
sys.path.append("/Users/skrishna/Documents/phd_codes/neurips_paper/")
sys.path.append("/Users/skrishna/Documents/phd_codes/neurips_paper/robustness_vs_counterfactuals/Recourse_Methods/AR")
sys.path.append("/Users/skrishna/Documents/phd_codes/neurips_paper/robustness_vs_counterfactuals/Recourse_Methods/Generative_Model")


# In[2]:



import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn as nn
import pickle as pkl
from numpy import linalg as LA


import ML_Models.data_loader as loader
# from utils import _get_input_subset
# from Recourse_Methods.gradient_methods import SCFE
# from utils import get_recourses, get_performance_measures


# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
#

# In[ ]:





# In[3]:


## Dataset Prep


from torchvision import  datasets, transforms
from torch.utils.data import DataLoader

data_name = "adult"
# data_name = "compas"
# data_name = "german"
n_starting_instances = 1200
compas_dict = {
        "data_path": '../Data_Sets/COMPAS/',
        "filename_train": 'compas-train.csv',
        "filename_test": 'compas-test.csv',
        "label": "risk",
        "task": "classification",
        "lr": 1e-3,
        "d": 6,
        "H1": 10,
        "H2": 10,
        "activFun": nn.Softplus(),
        "n_starting_instances": n_starting_instances
    }

german_dict = {
        "data_path": '../Data_Sets/German_Credit_Data/',
        "filename_train": 'german-train.csv',
        "filename_test": 'german-test.csv',
        "label": "credit-risk",
        "task": "classification",
        "lr": 1e-3,
        "d": 6,
        "H1": 10,
        "H2": 10,
    }

adult_dict = {
        "data_path": "../Data_Sets/Adult/",
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


data_meta_dictionaries = {
        "compas": compas_dict, 
        "adult": adult_dict, 
        "german":german_dict
    }
data_meta_info = data_meta_dictionaries[data_name]


dataset_test = loader.DataLoader_Tabular(path=data_meta_info["data_path"],
                                                 filename=data_meta_info["filename_test"],
                                                 label=data_meta_info["label"])
        
dataset_train = loader.DataLoader_Tabular(path=data_meta_info["data_path"],
                                                  filename=data_meta_info["filename_train"],
                                                  label=data_meta_info["label"])


column_names = pd.read_csv(data_meta_info["data_path"] + data_meta_info["filename_train"]).drop(data_meta_info["label"], axis=1).columns


# In[4]:


column_names


# In[5]:


# Data loader

train_loader = DataLoader(dataset_train, batch_size = 32, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size = 32, shuffle=False)

data = [i for i in train_loader]
num_input = len(data[0][0][0])


# In[6]:


class LinearModel(nn.Module):
    def __init__(self, num_input, output_layer):
        super(LinearModel, self).__init__()
        self.ff1 = nn.Linear(num_input, output_layer) 


    def forward(self, x):
        return self.ff1(x)
    


# In[7]:



model = torch.load("./models/{}_lr_model.pth".format(data_name))
model_robust = torch.load("./models/{}_lr_model_robust.pth".format(data_name))




# Recourse Method -1 
import torch
import numpy as np
from torch import nn
import datetime


class SCFE:
    
    def __init__(self, classifier, target_threshold: float = 0, _lambda: float = 10.0,
                 lr: float = 0.05, max_iter: int = 500, t_max_min: float = 0.5,
                 step: float = 0.10, norm: int = 1, optimizer: str = 'adam'):
        
        super().__init__()
        self.model_classification = classifier
        self.lr = lr
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.t_max_min = t_max_min
        self.norm = norm
        self.sigmoid = nn.Sigmoid()
        self.target_thres = target_threshold
        self._lambda = _lambda
        self.step = step
    
    def generate_counterfactuals(self, query_instance: torch.tensor, target_class: int = 1) -> torch.tensor:
        """
            query instance: the point to be explained
            target_class: Direction of the desired change. If target_class = 1, we aim to improve the score,
                if target_class = 0, we aim to decrese it (in classification and regression problems).
            _lambda: Lambda parameter (distance regularization) parameter of the problem
        """
        
        if target_class == 1:
            target_prediction = torch.tensor(1).float()
        else:
            target_prediction = torch.tensor(0).float()
        
        output = self._call_model(query_instance.reshape(1, -1))
        
        cf = query_instance.clone().requires_grad_(True)
        
        if self.optimizer == 'adam':
            optim = torch.optim.Adam([cf], self.lr)
        else:
            optim = torch.optim.RMSprop([cf], self.lr)
        
        # Timer
        t0 = datetime.datetime.now()
        t_max = datetime.timedelta(minutes=self.t_max_min)
        
        counterfactuals = []
        while not self._check_cf_valid(output, target_class):
            
            iter = 0
            distances = []
            all_loss = []
            
            while not self._check_cf_valid(output, target_class) and iter < self.max_iter:
                
                cf.requires_grad = True
                total_loss, loss_distance = self.compute_loss(self._lambda, cf,
                                                              query_instance,
                                                              target_prediction)
                
                optim.zero_grad()
                total_loss.backward(retain_graph=True)
                optim.step()
                
                output = self._call_model(cf)
                
                if self._check_cf_valid(output, target_class):
                    counterfactuals.append(cf.detach())
                    distances.append(loss_distance.clone().detach())
                    all_loss.append(total_loss.detach())
                
                iter = iter + 1
            
            # print('balance parameter: ', self._lambda)
            output = self._call_model(cf).reshape(1, -1).detach()
            if datetime.datetime.now() - t0 > t_max:
                # print('Timeout - No counterfactual explanation found')
                break
            # elif self._check_cf_valid(output, target_class):
                # print('Counterfactual explanation found')

            if self.step == 0.0:  # Don't search over lambdas
                break
            else:
                self._lambda -= self.step

        if not len(counterfactuals):
            print('No CE found')
            cf.detach_()
            return cf, None
        
        # Choose the nearest counterfactual
        counterfactuals = torch.stack(counterfactuals)
        distances = torch.stack(distances)
        distances = distances.detach()
        index = torch.argmin(distances)
        counterfactuals = counterfactuals.detach()

        ce_star = counterfactuals[index]
        distance_star = distances[index]
        
        
        return ce_star, distance_star
    
    def compute_loss(self, _lambda: float, cf_candidate: torch.tensor, original_instance: torch.tensor,
                     target: torch.tensor) -> torch.tensor:
        output = self._call_model(cf_candidate)
        # classification loss
        bce_loss = nn.BCEWithLogitsLoss()
#         print("Testing code : " , output, target)
        loss_classification = bce_loss(output, target)
        # distance loss
        loss_distance = torch.norm((cf_candidate - original_instance), self.norm)
        # full loss
        total_loss = loss_classification + _lambda * loss_distance
        return total_loss, loss_distance

    def _call_model(self, cf_candidate):
        output = self.model_classification(cf_candidate)[0]
        return output

    def _check_cf_valid(self, output, target_class):
        """ Check if the output constitutes a sufficient CF-example.
            target_class = 1 in general means that we aim to improve the score,
            whereas for target_class = 0 we aim to decrese it.
        """
        if target_class == 1:
            check = output >= self.target_thres
            return check
        else:
            check = output <= self.target_thres
            return check
        


# In[15]:


# Recourse Method 2
import Recourse_Methods.Generative_Model.model as model_vae
from numpy import linalg as LA

# Second class of counter-factual explanation methods         
class CCHVAE:

    def __init__(self, classifier, model_vae, target_threshold: float = 0,
                 n_search_samples: int = 1000, p_norm: int = 1,
                 step: float = 0.05, max_iter: int = 1000, clamp: bool = True):
        
        super().__init__()
        self.classifier = classifier
        self.generative_model = model_vae
        self.n_search_samples = n_search_samples
        self.p_norm = p_norm
        self.step = step
        self.max_iter = max_iter
        self.clamp = clamp
        self.target_treshold = target_threshold

    def hyper_sphere_coordindates(self, instance, high, low):
    
        """
        :param n_search_samples: int > 0
        :param instance: numpy input point array
        :param high: float>= 0, h>l; upper bound
        :param low: float>= 0, l<h; lower bound
        :param p: float>= 1; norm
        :return: candidate counterfactuals & distances
        """
    
        delta_instance = np.random.randn(self.n_search_samples, instance.shape[1])
        dist = np.random.rand(self.n_search_samples) * (high - low) + low  # length range [l, h)
        norm_p = LA.norm(delta_instance, ord=self.p_norm, axis=1)
        d_norm = np.divide(dist, norm_p).reshape(-1, 1)  # rescale/normalize factor
        delta_instance = np.multiply(delta_instance, d_norm)
        candidate_counterfactuals = instance + delta_instance
    
        return candidate_counterfactuals, dist

    def generate_counterfactuals(self, query_instance: torch.tensor, target_class: int = 1) -> torch.tensor:
        """
        :param instance: np array
        :return: best CE
        """  #

        # init step size for growing the sphere
        low = 0
        high = low + self.step

        # counter
        count = 0
        counter_step = 1
        query_instance = query_instance.detach().numpy()

        # get predicted label of instance
        self.classifier.eval()
        instance_label = 1 - target_class
        # vectorize z
        z = self.generative_model.encode_csearch(torch.from_numpy(query_instance).float()).detach().numpy()
        z_rep = np.repeat(z.reshape(1, -1), self.n_search_samples, axis=0)

        while True:
            count = count + counter_step
            if(count%300 == 0):
                print(count)

            if count > self.max_iter:
                candidate_counterfactual_star = np.empty(query_instance.shape[0], )
                candidate_counterfactual_star[:] = np.nan
                distance_star = np.nan
                print('No CE found')
                break

            # STEP 1 -- SAMPLE POINTS on hypersphere around instance
            latent_neighbourhood, _ = CCHVAE.hyper_sphere_coordindates(self, z_rep, high, low)

            x_ce = self.generative_model.decode_csearch(torch.from_numpy(latent_neighbourhood).float()).detach().numpy()

            if self.clamp:
                x_ce = x_ce.clip(-1, 1)

            # STEP 2 -- COMPUTE l1 & l2 norms
            if self.p_norm == 1:
                distances = np.abs((x_ce - query_instance)).sum(axis=1)
            elif self.p_norm == 2:
                distances = LA.norm(x_ce - query_instance, axis=1)
            else:
                print('Distance not defined yet')
            
            # counterfactual labels
            #print(x_ce.shape)
            y_candidate = torch.stack([torch.tensor([int(i[0])]) for i in self.classifier(torch.from_numpy(x_ce).float()).detach().numpy() > 0])
            
            #print(y_candidate.shape)

            indeces = np.where(y_candidate != instance_label)[0]
            candidate_counterfactuals = x_ce[indeces]
            candidate_dist = distances[indeces]

            if len(candidate_dist) == 0:  # no candidate found & push search range outside
                low = high
                high = low + self.step
            elif len(candidate_dist) > 0:  # certain candidates generated
                min_index = np.argmin(candidate_dist)
                candidate_counterfactual_star = candidate_counterfactuals[min_index]
                distance_star = np.abs(candidate_counterfactual_star - query_instance).sum()
                # print('CE found')
                break

        return torch.tensor(candidate_counterfactual_star), torch.tensor(distance_star)
    
    
    


# In[16]:


# Method 3 : Berk's recourse

import lime as lime
import lime.lime_tabular as lime_tabular

import torch
import pandas as pd
import numpy as np

from Recourse_Methods.AR.recourse.flipset import Flipset
from Recourse_Methods.AR.recourse.builder import ActionSet


class AR:
    def __init__(self, classifier, train_data: pd.DataFrame, func_class: str = 'ann', total_items: int = 50):
        self.classifier = classifier
        self.fun_class = func_class
        self.train_data = train_data
        self.total_items = 50

    def _build_lime(self, column_names, discretize_continuous: bool = False, sample_around_instance: bool = True):
        """
        Define a LIME explainer on dataset
        :param data: Dataframe with original train data
        :return: LimeExplainer
        """
        
        # Data preparation
        X = self.train_data.values
        lime_exp = lime.lime_tabular.LimeTabularExplainer(training_data=X,
                                                          discretize_continuous=discretize_continuous,
                                                          sample_around_instance=sample_around_instance, 
                                                          feature_names = column_names)
    
        return lime_exp
        
    def _get_lime_coefficients(self, instance : pd.DataFrame, column_names):
        
        """
        Actionable Recourse is not implemented for non-linear models and non binary categorical data.
        To mitigate the second issue, we have to use LIME to compute coefficients for our Black Box Model.
        :return: List of LIME-Explanations, intercept
        """
        # Prepare instance
        #print(instance.values, type(instance.values), instance.values.reshape((1, -1)), instance.index, instance.index.values)
        # Prepare instance
        inst_to_expl = instance #pd.DataFrame(instance.values.reshape((1, -1)),
                                    #columns=instance.index.values)
#         print("-----")
#         print("Whats the instance :", instance)
        inst_to_expl = instance #pd.DataFrame(instance.numpy().reshape((1, -1)),columns=column_names)

        lime_expl = self._build_lime(column_names)        
        def predict_fn_nn(x):
            return torch.cat((torch.tensor(self.classifier(torch.tensor(x, dtype = torch.float32)) < 0, dtype = torch.float32), 
                            torch.tensor(self.classifier(torch.tensor(x, dtype = torch.float32)) >= 0, dtype = torch.float32)), dim = 1).numpy()  
                             
                             
        # Prob. predictions
#         print("num_features : ", inst_to_expl.values.shape[1])
        explanations = lime_expl.explain_instance(np.squeeze(inst_to_expl.values),
                                                  predict_fn_nn,
                                                  num_features=inst_to_expl.values.shape[1])
#         print(explanations.as_list(), explanations.intercept)
        print(explanations.as_list())
        return explanations.as_list(), explanations.intercept[1]
    

    def generate_counterfactuals(self, column_names, query_instance: pd.DataFrame, target_class: int = 1) -> torch.tensor:
        
        action_set = ActionSet(X=self.train_data)
        
        # Actionable recourse is only defined on linear models
        # To use more complex models, they propose to use local approximation models like LIME
        
        if self.fun_class == 'ann':
            coeff, intercept = self._get_lime_coefficients(query_instance, column_names)
        else:
            coeff, intercept = self.classifier.get_coefficients()
            coeff = coeff[1].detach().numpy().tolist()
            intercept = intercept[1].detach().numpy().tolist()
        
        # Match LIME Coefficients with actionable recourse data
        # if LIME coef. is in ac_columns then use coefficient else 0
        print(coeff)
        ac_columns = self.train_data.columns
        rest_columns = [x for x in column_names if x not in ac_columns]
        
        # Turn top 10 LIME coefficients into list with coefficients containing all features
        # Features coefficients which are not in the top 10 are set to 0
        if self.fun_class == 'ann':
            coefficients = np.zeros(ac_columns.shape)
            for i, feature in enumerate(ac_columns):
                for t in coeff:
                    if t[0].find(feature) != -1:
                        coefficients[i] += t[1]
        else:
            coefficients = coeff
            
        # Align action set to coefficients
        action_set.set_alignment(coefficients=coefficients)
        
        # Build counterfactuals
        rest_df = query_instance[rest_columns].values.reshape((1, -1))
        rest_df = pd.DataFrame(rest_df, columns=rest_columns)
        inst_for_ac = query_instance[ac_columns].values.reshape((1, -1))
        inst_for_ac = pd.DataFrame(inst_for_ac, columns=ac_columns)
        
        fb = Flipset(
            x=inst_for_ac.values,
            action_set=action_set,
            mip_cost_type='l2',
            coefficients=coefficients,
            intercept=intercept
        )
        
        # Fit AC and build counterfactual
        fb_set = fb.populate(enumeration_type='distinct_subsets', total_items=self.total_items)
        actions_flipset = fb_set.actions
        actions_flipset = sorted(actions_flipset, key=lambda x: np.sqrt(np.dot(np.array(x), np.array(x))))
        last_object = len(actions_flipset) - 1
        for idx, action in enumerate(actions_flipset):
            counterfactual = inst_for_ac.values + action
            counterfactual = pd.DataFrame(counterfactual, columns=ac_columns)
            counterfactual[rest_columns] = rest_df[rest_columns]
            counterfactual = counterfactual[
                query_instance.columns]  # Arrange instance and counterfactual in same column order
                
        distance_l1 = LA.norm(counterfactual.values - query_instance.values, 1)
        distance_l2 = LA.norm(counterfactual.values - query_instance.values, 2)
        
        return counterfactual, distance_l1, distance_l2

    
    


# In[17]:


def _get_input_subset(model, inputs: torch.tensor,
                      subset_size: int = 100,
                      decision_threshold: float = 0) -> torch.tensor:
    
    """
    Get negatively classified inputs & return their predictions
    """
    
    yhat = (model(inputs) > decision_threshold) * 1
    check = (model(inputs) < decision_threshold).detach().numpy()
    selected_indices = np.where(check)[0]
    input_subset = inputs[selected_indices]
    predicted_label_subset = yhat[selected_indices]
    return input_subset[0:subset_size, :], predicted_label_subset[0:subset_size]


# In[18]:


# This method makes the "Manifold assumption" and uses random search in latent space

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
subset_size = 100

        
testloader = DataLoader(dataset_test,
                        batch_size=data_meta_info["n_starting_instances"],
                        shuffle=False)

data_iter = iter(testloader)
inputs, labels, indeces = data_iter.next()
inputs = inputs.to(device).float()

inputs, predicted_classes = _get_input_subset(model, inputs, subset_size=subset_size, decision_threshold=0)
inputs = inputs.numpy()
inps_base = pd.DataFrame(inputs)
inps_base.columns = column_names

ar = AR(classifier=model, train_data=inps_base)


testloader = DataLoader(dataset_test,
                        batch_size=data_meta_info["n_starting_instances"],
                        shuffle=False)

data_iter = iter(testloader)
inputs, labels, indeces = data_iter.next()
inputs = inputs.to(device).float()
inputs, predicted_classes = _get_input_subset(model_robust, inputs, subset_size=subset_size, decision_threshold=0)
inputs = inputs.numpy()
inps_ro = pd.DataFrame(inputs)
inps_ro.columns = column_names



ar2 = AR(classifier=model_robust, train_data=inps_ro)


# In[19]:


column_names


# In[23]:


# Generate counterfactuals for  

# inputs = pd.DataFrame(inputs, columns = column_names)

# counterfactual, distance_l1, distance_l2 = ar.generate_counterfactuals(
#                     column_names, query_instance= inputs.loc[[0]], target_class=1)


print(inps_ro.shape)


distances_base_ar = [ar.generate_counterfactuals(
                    column_names, query_instance= inps_base.loc[[i]], target_class=1) for i in range(1)] #range(inps_base.shape[0])]


distances_ro_ar = [ar2.generate_counterfactuals(
                    column_names, query_instance= inps_ro.loc[[i]], target_class=1) for i in range(1)] # range(inps_base.shape[0])]





