# Utils
import torch
import numpy as np
from torch import nn
from numpy import linalg as LA
import datetime


class REVISE:

    def __init__(self, classifier, model_vae, optimizer: str = "adam", max_iter: int = 750,
                 target_threshold: float = 0.5, t_max_min: float = 0.5, _lambda: float = 10.0,
                 lr: float = 0.05, norm: int = 1, step: float = 0.05):
        
        super().__init__()
        self.model_classification = classifier
        self.model_vae = model_vae
        self.lr = lr
        self.norm = norm
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.target_treshold = target_threshold
        self._lambda = _lambda
        self.t_max_min = t_max_min
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

        query_instance = query_instance.clone().detach()
        z = self.model_vae.encode_csearch(query_instance).clone().detach()
        z.requires_grad = True
        
        if self.optimizer == "adam":
            optim = torch.optim.Adam([z], self.lr)
        else:
            optim = torch.optim.RMSprop([z], self.lr)
            
        # Timer
        t0 = datetime.datetime.now()
        t_max = datetime.timedelta(minutes=self.t_max_min)
        
        counterfactuals = []  # all possible counterfactuals
        distances = []        # distance of the possible counterfactuals from the initial value

        # set for now: will be 1 in our setting
        output = torch.tensor(1) - target_class

        while not self._check_cf_valid(output, target_class):
    
            it = 0
            distances = []

            while not self._check_cf_valid(output, target_class) and it < self.max_iter:
    
                cf = self.model_vae.decode_csearch(z)
                output = self.model_classification(cf)
                predicted = output[0] > self.target_treshold
                
                if predicted == target_prediction:
                    counterfactuals.append(cf)
                    
                z.requires_grad = True
                total_loss, distance = self.compute_loss(cf_proposal=cf,
                                                         query_instance=query_instance,
                                                         target=target_prediction,
                                                         _lambda=self._lambda)
                
                optim.zero_grad()
                total_loss.backward()
                optim.step()
                cf.detach_()

                if self._check_cf_valid(output, target_class):
                    counterfactuals.append(cf.detach())
                    distances.append(distance.clone().detach())

                it = it + 1

            output = self.model_classification(cf)
            if datetime.datetime.now() - t0 > t_max:
                print('Timeout')
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
            return cf, torch.tensor(np.nan)

        # Choose the nearest counterfactual
        counterfactuals = torch.stack(counterfactuals)
        
        distances = torch.stack(distances)
        distances = distances.detach()
        index = torch.argmin(distances)
        counterfactuals = counterfactuals.detach()

        ce_star = counterfactuals[index]
        distance_star = distances[index]

        return ce_star, distance_star

    def compute_loss(self, cf_proposal, query_instance, target, _lambda):

        loss_function = nn.BCELoss() 
        output = self.model_classification(cf_proposal)[0]
        loss_classification = loss_function(output, target)
        loss_distance = torch.norm((cf_proposal - query_instance), self.norm)
        total_loss = loss_classification + _lambda * loss_distance
        return total_loss, loss_distance
    
    def _check_cf_valid(self, output, target_class):
        """ Check if the output constitutes a sufficient CF-example.
            target_class = 1 in general means that we aim to improve the score,
            whereas for target_class = 0 we aim to decrese it.
        """
        if target_class == 1:
            check = output >= self.target_treshold
            return check
        else:
            check = output <= self.target_treshold
            return check


class CCHVAE:

    def __init__(self, classifier, model_vae, target_threshold: float = 0.5,
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
            y_candidate = np.argmax(self.classifier(torch.from_numpy(x_ce).float()).detach().numpy(), axis=1)

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

    

