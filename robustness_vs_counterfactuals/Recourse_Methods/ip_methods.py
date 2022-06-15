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

    def _build_lime(self, discretize_continuous: bool = False, sample_around_instance: bool = True):
        """
        Define a LIME explainer on dataset
        :param data: Dataframe with original train data
        :return: LimeExplainer
        """
        
        # Data preparation
        X = self.train_data.values
        lime_exp = lime.lime_tabular.LimeTabularExplainer(training_data=X,
                                                          discretize_continuous=discretize_continuous,
                                                          sample_around_instance=sample_around_instance)
    
        return lime_exp
        
    def _get_lime_coefficients(self, instance):
        
        """
        Actionable Recourse is not implemented for non-linear models and non binary categorical data.
        To mitigate the second issue, we have to use LIME to compute coefficients for our Black Box Model.
        :return: List of LIME-Explanations, intercept
        """
        print(instance.values, type(instance.values))
        # Prepare instance
        inst_to_expl = pd.DataFrame(instance.values.reshape((1, -1)),
                                    columns=instance.index.values)

        lime_expl = self._build_lime()
        
        # Prob. predictions
        explanations = lime_expl.explain_instance(np.squeeze(inst_to_expl.values),
                                                  self.classifier.predict,
                                                  num_features=inst_to_expl.values.shape[1])
        
        return explanations.as_list(), explanations.intercept[1]

    def generate_counterfactuals(self, query_instance: pd.DataFrame, target_class: int = 1) -> torch.tensor:
        
        action_set = ActionSet(X=self.train_data)
        
        # Actionable recourse is only defined on linear models
        # To use more complex models, they propose to use local approximation models like LIME
        
        if self.fun_class == 'ann':
            coeff, intercept = self._get_lime_coefficients(query_instance)
        else:
            coeff, intercept = self.classifier.get_coefficients()
            coeff = coeff[1].detach().numpy().tolist()
            intercept = intercept[1].detach().numpy().tolist()
        
        # Match LIME Coefficients with actionable recourse data
        # if LIME coef. is in ac_columns then use coefficient else 0
        ac_columns = self.train_data.columns
        rest_columns = [x for x in query_instance.columns if x not in ac_columns]
        
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
                
        return counterfactual
